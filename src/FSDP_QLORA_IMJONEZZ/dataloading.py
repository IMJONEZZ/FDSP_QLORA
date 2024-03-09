#Packages needed: llama-recipies fastcore --extra-index-url https://download.pytorch.org/whl/test/cu121
# General
import torch, os, gc, time, safetensors, copy, math, types
import functools
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_linear_schedule_with_warmup
import bitsandbytes as bnb
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from safetensors.torch import save_file
from tqdm.auto import tqdm
from typing import List, Dict

# Argument parsing
from fastcore.script import call_parse, bool_arg, Param

try:
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
except ImportError:
    HQQLinear = None
    pass

# Torch + distributed training
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# FSDP
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Model loading
from bitsandbytes.nn import Linear4bit, Params4bit
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from peft import get_peft_model, LoraConfig, TaskType
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from fastcore.parallel import parallel

#PEFT
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

# For different model types, we'll want to import the right class for the
# check_fn in activation checkpointing (LlamaDecoderLayer for llama models for example)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb
except ImportError:
    pass

# DATASET + DATALOADERS (modified from llama recipes)
# Formatting prompts in alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# Dataset class
class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, style="alpaca"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.style = style

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        if self.style == "guanaco":
            prompt = self.dataset[index]["text"].split("### Assistant: ")[0]
            example = self.dataset[index]["text"]
        elif self.style == "qna":
            prompt_template = "###Context:\n{context}\n###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample['answer']
        else: # Alpaca
            ann = self.dataset[index]
            if ann.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            example = prompt + ann["output"]

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

# And to get the dataloader
def get_dataloader(tokenizer:PreTrainedTokenizerFast, args:Dict):
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import Dataset, load_dataset

    # Load the source dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")['train']
    elif args["dataset"] == "alpaca_sample":
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:512]")
    elif args["dataset"] == "dummy":
        dataset = Dataset.from_dict({
            'instruction': ["instruction"]*512,
            'input': ["input"]*512,
            'output': ["output"*10000]*512} # A long output to test memory usage (gets truncated)
        )
    elif args["dataset"] == "guanaco":
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    elif args["dataset"] == "sql":
        dataset = load_dataset("knowrohit07/know_sql")['validation']
        dataset = dataset.shuffle(seed=args["seed"])
        dataset = dataset.select(range(1000,len(dataset)))

    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    dataset = dataset.select(range(0, len(dataset)-len(dataset)%(args["batch_size"]*args["gradient_accumulation_steps"])))

    # # Create the InstructionDataset
    if args["dataset"] == "guanaco":
        dataset = InstructionDataset(dataset, tokenizer, style="guanaco")
    elif args["dataset"] == "sql":
        dataset = InstructionDataset(dataset, tokenizer, style="qna")
    else: # (w/ alpaca prompt formatting)
        dataset = InstructionDataset(dataset, tokenizer, style="alpaca")

    # Collate function
    def collate_fn(batch, with_attention_mask=False):
        # To list of tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :args["context_length"]]
        if with_attention_mask:
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :args["context_length"]]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[:, :args["context_length"]]
        # Return dict
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset, seed=args["seed"])

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_fn, sampler=sampler)

    return dataloader


# LR scheduler.
def _get_cosine_one_cycle_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_fraction = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = (1 - min_lr_fraction)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return (math.cos(math.pi * progress)+1) * 0.5 * scale_term + min_lr_fraction

def get_cosine_one_cycle_scheduler(optimizer:optim.Optimizer, num_warmup_steps:int, num_training_steps:int, min_lr_fraction:float=0.1):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_lr_scheduler(optimizer:optim.Optimizer, dataloader:DataLoader, gradient_accumulation_steps:int, args:Dict):
    """Returns linear, cosine, or constant learning rate scheduler"""
    num_training_steps = args['num_epochs'] * len(dataloader) // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * 0.1)
    if args['lr_scheduler'] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args['lr_scheduler'] == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1)
    elif args['lr_scheduler'] == "constant":
        lr_scheduler = None
    else:
        raise NotImplementedError(f"{args['lr_scheduler']} LR scheduler not implemented yet")
    return lr_scheduler, num_training_steps


# Optimizer
def get_optimizer(model:nn.Module, args:Dict):
    """Returns an optimizer. We can add more options here if needed."""
    if args["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args['lr'], betas=(0.9,0.95),
                                 eps=1e-5, weight_decay=args['wd'])
    else:
        raise ValueError("Invalid optimizer")


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy:bool=False):
    if custom_policy:
        def lambda_policy_fn(module):
            # LORA trainable layers.
            return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module))
    else:
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, tuple(LLAMA_ATTENTION_CLASSES.values()))

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, LlamaMLP)

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_layer_name = LlamaDecoderLayer
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
        ),
    )
    policies=[lambda_policy, transformer_wrap_policy]
    if custom_policy:
        policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)
