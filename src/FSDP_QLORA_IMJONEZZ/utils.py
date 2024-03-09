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
from torch import nn, Tensor
from bitsandbytes.nn import Linear4bit, Params4bit

try:
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
except ImportError:
    HQQLinear = None
    pass


# Utilities related to model loading
def replace_linear(model:nn.Module, linear_replacement:nn.Module, quant_config:dict|None=None,
                   skip_modules:List[str]=["lm_head"], **kwargs):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, quant_config, skip_modules, **kwargs)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            if issubclass(linear_replacement, Linear4bit):
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    **kwargs
                )
            elif issubclass(linear_replacement, HQQLinear):
                model._modules[name] = linear_replacement(module, quant_config, **kwargs)
            else:
                raise ValueError(f"Unsupported linear replacement: {type(linear_replacement)}")
    return model


def setup_quantized_meta_for_peft(model:nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""
    def temp_to_method(self, *args, **kwargs):
        return self
    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)

def setup_quantized_peft_meta_for_training(model:nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, '_orig_to'):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None

def load_and_quantize(module:nn.Module, name:str, value:Tensor, device:torch.device=None, dtype:torch.dtype=None,
                      skip_names:list[str]=[], is_meta_rank:bool=False, low_memory:bool=True, verbose:bool=False, quant_method:str='bnb'):
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if low_memory=True or "meta" if is_meta_rank=True.
    """
    def place_on_device(value):
        if is_meta_rank:
            device = 'meta'
        elif low_memory:
            device = 'cpu'
        return value.to(device=device, dtype=dtype)

    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition('.')
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as e:
        print(f"Module {module_key} not found:\n{e}")
        return

    try:
        if quant_method=='bnb':
            param = submodule.get_parameter(value_key)
            if isinstance(param, Params4bit):
                # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                # shape as the quantized Params4bit with an initialized quant_state. However,
                # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                value = type(param)(value.to(device=device, dtype=dtype).data, **param.__dict__).cuda(device)
                if is_meta_rank:
                    value = type(param)(value.data.to("meta"), **value.__dict__)
                elif low_memory:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)
        elif quant_method=='hqq':
            if isinstance(submodule, HQQLinear):
                if value_key == "weight":
                    # Like `Params4bit`, this workaround quantizes `HQQLinear`` per device so the quantization
                    # meta dictionary is created on all ranks, before converting to meta on non-rank 0.
                    submodule.linear_layer.to_empty(device=device)
                    submodule.linear_layer.weight.data.copy_(value.to(device=device, dtype=dtype))
                    submodule.initialize()

                    if is_meta_rank:
                        setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("meta")))
                    elif low_memory:
                        setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("cpu")))
                    submodule.in_gpu = False

                if value_key == "bias":
                    raise ValueError("Bias not supported in HQQLinear yet!")
            else:
                param = submodule.get_parameter(value_key)
                value = type(param)(place_on_device(value).data)

    except AttributeError:
        # it's a buffer
        value = place_on_device(value)
        pass
    if HQQLinear is None or not isinstance(submodule, HQQLinear):
        setattr(submodule, value_key, value)

# Main function, run on each process
def fsdp_main(local_rank:int, world_size:int, args:Dict):
    print_func = tqdm.write if args["log_to"] == 'tqdm' else print

    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    if 'SLURM_PROCID' in os.environ:
        # assumes same number of GPUs per node.
        rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
    else:
        rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Start logging
    logger = Logger(args, log_to=args["log_to"], project_name=args["project_name"],
                    entity=args["entity"], group=args["group"], name=args["name"], rank=rank)

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16
    mp_policy = None
    load_param_skip_names = []
    if args["precision"] == "bf16":
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        load_param_skip_names = ['inv_freq']
    else:
        raise ValueError("Invalid precision")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # Set up dataloader
    dataloader = get_dataloader(tokenizer, args)


    # Create model
    cfg = None
    attn_impl = "sdpa" # torch 2.2 sdpa uses flash attn 2
    print("Creating model", rank)
    if args["train_type"] in ["full", "lora", "custom_lora"]:
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            model = AutoModelForCausalLM.from_pretrained(
                args["model_name"],
                use_cache=False,
                torch_dtype=torch_dtype,
                _attn_implementation=attn_impl
            )
            dtype = torch_dtype if args["precision"] == "bf16" else None
            model.to(dtype=dtype, device="cpu" if args["low_memory"] else rank)
        else:
            cfg = AutoConfig.from_pretrained(args["model_name"])
            cfg.use_cache = False
            cfg._attn_implementation = attn_impl
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch_dtype)
            if args["precision"] == "bf16":
                model.to(torch_dtype)
    elif args["train_type"] in ["qlora", "custom_qlora", "hqq_lora"]: # Our custom loading
        cfg = AutoConfig.from_pretrained(args["model_name"])
        cfg.use_cache = False
        cfg._attn_implementation = attn_impl

        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            if args["train_type"] in ["hqq_lora"]:
                # TODO: Tune BaseQuantizeConfig.
                quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True,
                                                  quant_scale=True, offload_meta=True, view_as_float=True)
                model.model = replace_linear(model.model, HQQLinear, quant_config, device=rank,
                                             compute_dtype=compute_dtype, del_orig=True, initialize=False)
                HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
            else:
                model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                             quant_type='nf4', quant_storage=torch_dtype)
        model.is_loaded_in_4bit = True

        # Grab the safetensors files that hold the weights
        try:
            idx = hub.cached_file(args["model_name"], SAFE_WEIGHTS_INDEX_NAME)
            files, _ = hub.get_checkpoint_shard_files(args["model_name"], idx)
        except OSError:
            try:
                # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                files = []
                files.append(hub.cached_file(args["model_name"], SAFE_WEIGHTS_NAME))
            except OSError as e:
                # This means the model probably doesn't have a safetensors file
                raise e

        # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
        # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
        def load_and_quantize_parallel(name_param, model, **kwargs):
            name, param = name_param
            load_and_quantize(model, name, param, **kwargs)

        print("Loading model", rank)
        param_count = sum((p.numel() for n,p in model.named_parameters()))
        if rank == 0 and args['verbose']:
            print_func(f"Total model params: {param_count}")
        start = time.time()
        for filename in files:
            weights = safetensors.torch.load_file(filename)
            quant_method = "hqq" if args["train_type"] in ["hqq_lora"] else "bnb"
            devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
            left = int(os.cpu_count()/torch.cuda.device_count())
            right = int(8 * (devprops.total_memory/1e9/40) * (70/(param_count/1e9)))
            n_workers = min(left, right)
            if rank == 0 and args['verbose']:
                print_func(f"Using n_workers: {n_workers} for loading")
            parallel(load_and_quantize_parallel, weights.items(), n_workers=n_workers, threadpool=True,
                     model=model, dtype=torch_dtype, device=local_rank, skip_names=load_param_skip_names,
                     is_meta_rank=(args["low_memory"] and rank!=0), verbose=args["verbose"], quant_method=quant_method)
        if rank == 0 and args["verbose"]:
            print(f"Loaded model weights in {time.time()-start:.3f} seconds")

    print("Model created", rank, f"{torch.cuda.memory_allocated(local_rank)/1e9:.3f} GB")


    # PEFT setup (LoRA and QLoRA)
    if args["train_type"] in ["lora", "qlora"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args["lora_rank"],
            lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"],
            target_modules=args["lora_target_modules"],
        )
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank!=0 and args["low_memory"]:
            setup_quantized_meta_for_peft(model)

        model = get_peft_model(model, peft_config)

        if rank==0:
            model.print_trainable_parameters()
        elif args['low_memory']:
            # And then setup_quantized_peft_meta_for_training sets quant_state.to back to normal
            setup_quantized_peft_meta_for_training(model)
    elif args["train_type"] in ["custom_qlora", "custom_lora", "hqq_lora"]:
        # Create LORA layers.
        for name, _ in model.named_modules():
            module_key, _, value_key = name.rpartition('.')
            if value_key in args['lora_target_modules']:
                m = model.get_submodule(name)
                qlora_layer = LORA(m, args["lora_rank"], args["lora_alpha"], args["lora_dropout"])
                parent_module = model.get_submodule(module_key)
                setattr(parent_module, value_key, qlora_layer)
        for n,p in model.named_parameters():
            if any([lora_name in n for lora_name in ['lora_AB', 'lora_A', 'lora_B']]):
                p.requires_grad = True
                if args['verbose']:
                    print("Trainable LORA layer", n)
            else:
                p.requires_grad = False

        print("LoRA layers added", rank, f"{torch.cuda.memory_allocated(local_rank)/1e9:.3f} GB")

    logger.log({"memory_after_model_creation": torch.cuda.memory_allocated(local_rank)}, rank)


    # Wrap model with llama-recipies or custom LoRA policy
    my_auto_wrap_policy = get_wrapping_policy(args["train_type"] in ["custom_qlora", "hqq_lora"])

    print("Wrapping model w/ FSDP", rank)
    if args["sharding_strategy"] == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args["sharding_strategy"] == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args["sharding_strategy"] == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args["sharding_strategy"] == "hybrid_full_shard":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args["sharding_strategy"] == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError("Invalid FSDP sharding strategy")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"],
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and args["low_memory"]) else None, # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    print("Wrapped model", rank, f"{torch.cuda.memory_allocated(local_rank)/1e9:.3f} GB")
    logger.log({"memory_after_model_wrap": torch.cuda.memory_allocated(local_rank)}, rank)


    # Synchronize at the start
    dist.barrier()

    # model = torch.compile(model)

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        if args['reentrant_checkpointing']:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT if args['reentrant_checkpointing'] else CheckpointImpl.NO_REENTRANT,

        )

        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if args["use_activation_cpu_offload"]:
        print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 and args['verbose']:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")


    # Create the optimizer
    optimizer = get_optimizer(model, args)

    # LR scheduler.
    gradient_accumulation_steps = max(1, args['gradient_accumulation_steps'])
    lr_scheduler, num_training_steps = get_lr_scheduler(optimizer, dataloader, gradient_accumulation_steps, args)

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}")


    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None


    if rank == 0:
        print("Total Training Steps:", num_training_steps)
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    init_start_event.record()
    log_loss, log_lr = 0.0, -1
    # Reset peak memory to track that
    torch.cuda.reset_peak_memory_stats(local_rank)
    for epoch in range(args['num_epochs']):
        update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
        model.train()
        ddp_loss = torch.zeros(2).to(local_rank)

        for batch_idx, batch in enumerate(dataloader):
            accumulate_grads = (batch_idx+1) % gradient_accumulation_steps == 0

            # Prevent gradient syncing until update step if using no_sync option.
            # Documentation states this should only be used on the root FSDP instance
            # We assume this is a one-node setup
            if args['no_sync'] and not accumulate_grads:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            # Start logging memory (first iter) if requested
            if batch_idx==0 and rank == 0 and epoch == 0 and args['profile_memory']:
                torch.cuda.memory._record_memory_history()

            # Log memory usage
            if batch_idx == 4 and epoch == 0:
                logger.log({"memory_before_forward": torch.cuda.memory_allocated(local_rank)/1e9}, rank)

            # Forward pass
            with sync_context:
                with autocast:
                    output = model(
                        batch['input_ids'].to(local_rank),
                        labels=batch['labels'].to(local_rank),
                        attention_mask=None,
                    )
                    loss = output.loss

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Log memory usage
                if batch_idx == 4 and epoch == 0:
                    logger.log({"memory_after_forward": torch.cuda.memory_allocated(local_rank)/1e9}, rank)

                # Backward pass
                if scale_grads:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Record loss
            bs = batch['input_ids'].shape[0]
            ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
            ddp_loss[1] += bs

            # Step the optimizer (w/ gradient accumulation)
            if accumulate_grads:
                if args['apply_gradient_clipping'] and (args['grad_norm'] is not None):
                    model.clip_grad_norm_(args['grad_norm'], norm_type=2.0)
                if scale_grads:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # avoid overhead when lr is constant.
                if lr_scheduler is not None:
                    lr_scheduler.step()
                progress_bar.update(1)

            # Log memory usage after backwards
            if batch_idx == 4 and epoch == 0:
                logger.log({"memory_after_backward": torch.cuda.memory_allocated(local_rank)/1e9}, rank)

            # Delete the output so more memory frees up before the next forward pass
            output = None
            loss = None

            # Stop logging memory (first iter)
            if batch_idx==0 and rank == 0 and epoch == 0 and args['profile_memory']:
                torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                torch.cuda.memory._record_memory_history(enabled=None) # Stop recording

            # Log loss every gradient update steps
            if accumulate_grads:
                dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                if rank == 0:
                    log_loss = ddp_loss[0] / ddp_loss[1]
                    if lr_scheduler is not None:
                        log_lr = lr_scheduler.get_last_lr()[0]
                    else:
                        log_lr = args["lr"]
                    update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                    if args["log_to"] == 'wandb':
                        logger.log({"loss": log_loss, "lr": log_lr}, rank)
                ddp_loss = torch.zeros(2).to(local_rank)

        # Print + log peak memory usage for the whole first step of training
        if epoch == 0 and rank == 0:
            peak_memory = torch.cuda.max_memory_allocated(local_rank)
            if args["verbose"]:
                print_func(f"Peak memory usage (training): {peak_memory/1e9:.2f}GB", rank)
            if args["log_to"] == 'wandb':
                logger.log({"memory_peak": peak_memory}, rank)

    # Synchronize at the end and record time
    init_end_event.record()
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print("Finished training", rank)

    # Print time and model
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {time_taken} sec")
        logger.log({"time_taken": time_taken}, rank)

    # End logging
    logger.finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    # HQQLinear custom state_dict() method causes issues when saving.
    # Model is saved fine when `state_dict()` method is removed.
    # Non param/buffer types are not saved with FSDP.
    # It might be better to just save the trained lora layers.
    # summon_full_params on lora layers and save.
    if args["save_model"]:
        if rank == 0:
            os.makedirs(args["output_dir"], exist_ok=True)
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if args["train_type"] in ["custom_lora", "custom_qlora", "hqq_lora"]:
            cpu_state_dict = {}
            trainable_modules = [(n,m) for n,m in model.named_modules() if n.endswith('lora_AB')]
            for prefix, module in trainable_modules:
                prefix = (prefix.replace("_fsdp_wrapped_module.", "")
                                .replace("_checkpoint_wrapped_module.", "")
                                .replace("_offload_wrapped_module.", ""))
                with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state_dict = {**cpu_state_dict, **{f"{prefix}.{k}":v for k,v in module.state_dict().items()}}
                dist.barrier()
                torch.cuda.synchronize()
            if rank==0:
                print_func("Saving trained LoRA weights.")
                save_file(cpu_state_dict, os.path.join(args["output_dir"], "model_state_dict.safetensors"))
                print_func("Done", rank)
        else:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
                if rank==0:
                    print_func("Saving full model weights.")
                    save_file(cpu_state_dict, os.path.join(args["output_dir"], "model_state_dict.safetensors"))
                    print_func("Done", rank)

    dist.barrier() # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()