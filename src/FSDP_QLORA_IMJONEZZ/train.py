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

#Local
from logging import Logger
from utils import replace_linear, setup_quantized_meta_for_peft, setup_quantized_peft_meta_for_training, load_and_quantize, fsdp_main
from dataloading import PROMPT_DICT, InstructionDataset, get_dataloader, _get_cosine_one_cycle_lr_lambda, get_cosine_one_cycle_scheduler, get_lr_scheduler, get_optimizer, get_wrapping_policy


class FSDP_QLORA:
    def __init__(self,
                world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
                train_type: str = "qlora", # "full", "lora", "qlora", or "custom_qlora"
                batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
                context_length: int = 512, # Max length of input sequence (in tokens)
                gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
                num_epochs: int = 1, # How many epochs of training to do
                dataset: str = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
                sharding_strategy: str = "full_shard", # Sharding strategy for FSDP
                use_gradient_checkpointing: bool_arg = True, # Use FSDP's activation checkpointing
                reentrant_checkpointing: bool_arg = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
                use_cpu_offload: bool_arg = True, # Use FSDP's CPU offloading
                use_activation_cpu_offload: bool_arg = False, # Use FSDP's activation CPU offloading
                low_memory: bool_arg = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
                no_sync: bool_arg = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
                precision: str = "bf16", # Training precision. autocast precisions use mixed precision
                model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                save_model: bool_arg = False, # Save the resulting model
                output_dir: str = "output", # Output directory to save the final model to
                lora_rank: int = 64, # LoRA rank for lora/qlora
                lora_alpha: int = 16, # LoRA alpha for lora/qlora
                lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
                lora_target_modules: str = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
                verbose: bool_arg = False, # Whether to print extra info for debugging
                lr: float = 1e-5, # Learning rate
                apply_gradient_clipping: bool_arg = False, # Apply gradient norm clipping
                grad_norm: float = 0.3, # Gradient norm clipping
                wd: float = 0.1, # Weight decay
                profile_memory: bool_arg = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
                optimizer: str = "adamw", # Optimizer
                lr_scheduler: str = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
                log_to: str = "tqdm", # Where to log output
                master_addr: str = "localhost", # For distributed training
                master_port: str = "12355", # For distributed training, must be the same for all processes
                seed: int = 42, # Random seed
                project_name: str = "fsdp_qlora", # For wandb logging
                name: str = None, # For wandb logging
                group: str = None, # For wandb logging
                entity: str = None, # For wandb logging
                 ):
        world_size = world_size if world_size != -1 else torch.cuda.device_count()
        print(f"World size: {world_size}")

        if lora_target_modules == "all":
            lora_target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
        elif lora_target_modules.lower() == "default":
            lora_target_modules = None

        if precision in ["bf16", "bf16_autocast", "bf16_buffers_autocast"] and not torch.cuda.is_available():
            raise ValueError("dummy you don't have cuda configured")
        
        if use_cpu_offload and gradient_accumulation_steps > 1:
            no_sync = True

        elif no_sync and gradient_accumulation_steps == 1:
            no_sync = False

        if train_type in ["hqq_lora"] and HQQLinear is None:
            raise ValueError("dummmmmmmyyyyyy, you don't have HQQ configured")

        args = {
            "train_type":train_type,
            "batch_size":batch_size,
            "context_length":context_length,
            "gradient_accumulation_steps":gradient_accumulation_steps,
            "num_epochs":num_epochs,
            "dataset":dataset,
            "sharding_strategy":sharding_strategy,
            "use_gradient_checkpointing":use_gradient_checkpointing,
            "reentrant_checkpointing":reentrant_checkpointing,
            "use_cpu_offload":use_cpu_offload,
            "use_activation_cpu_offload":use_activation_cpu_offload,
            "low_memory":low_memory,
            "no_sync":no_sync,
            "precision":precision,
            "model_name":model_name,
            "save_model":save_model,
            "output_dir":output_dir,
            "lora_rank":lora_rank,
            "lora_alpha":lora_alpha,
            "lora_dropout":lora_dropout,
            "lora_target_modules":lora_target_modules,
            "verbose":verbose,
            "lr":lr,
            "apply_gradient_clipping":apply_gradient_clipping,
            "grad_norm":grad_norm,
            "wd":wd,
            "profile_memory":profile_memory,
            "optimizer":optimizer,
            "lr_scheduler":lr_scheduler,
            "log_to":log_to,
            "master_addr":master_addr,
            "master_port":master_port,
            "seed":seed,
            "project_name":project_name,
            "name":name,
            "group":group,
            "entity":entity,
        }

        def train_qlora(self):
            mp.spawn(
                fsdp_main,
                args=(world_size, args),
                nprocs=torch.cuda.device_count(),
                join=True
            )