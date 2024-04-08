

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration for training a GPT-2 (124M) on OpenWebText dataset
# Output directories and logging
output_directory = 'output'
evaluation_interval = 2000
logging_interval = 1
evaluation_steps = 200
evaluation_mode = False  # Set True to exit after first evaluation
save_checkpoint_every_eval = True  # Always save a checkpoint after evaluation
initialization_mode = 'scratch'  # Options: 'scratch', 'resume', 'gpt2*'
# Weights and Biases logging
wandb_enabled = False  # Disabled by default
wandb_project_name = 'openwebtext_project'
wandb_run_identifier = 'gpt2_training_run'
# Data loading
dataset_name = 'openwebtext'
grad_accumulation_steps = 5 * 8  # Used to simulate larger batch sizes
micro_batch_size = 12  # Micro-batch size, real batch size will be multiplied by grad_accumulation_steps
context_length = 1024
# Model configuration
num_layers = 12
num_attention_heads = 12
embedding_size = 768
dropout_rate = 0.0  # For pretraining, 0 is good. For fine-tuning, try 0.1+
use_bias_in_model = False  # Controls bias in LayerNorm and Linear layers
# Optimizer (AdamW) settings
base_learning_rate = 6e-4
total_training_iterations = 600000
weight_decay_rate = 1e-1
beta1 = 0.9
beta2 = 0.95
gradient_clip_value = 1.0  # Clip gradients if non-zero, else disable clipping
# Learning rate decay settings
enable_lr_decay = True  # Whether to decay the learning rate
warmup_iterations = 2000
lr_decay_total_steps = 600000  # Should match total_training_iterations
minimum_learning_rate = 6e-5  # Min learning rate (suggested to be 10% of base_learning_rate)
# DDP (Distributed Data Parallel) settings
ddp_backend = 'nccl'  # 'nccl', 'gloo', etc.
# System and precision
selected_device = 'cuda'
tensor_dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
enable_compilation = True  # Use PyTorch 2.0 compilation for speedup
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Override from the command line or config file
config = {k: globals()[k] for k in config_keys}  # Save configuration for logging purposes
# -----------------------------------------------------------------------------

# DDP initialization and settings
distributed_training = int(os.environ.get('RANK', -1)) != -1
if distributed_training:
    init_process_group(backend=ddp_backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    selected_device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(selected_device)
    is_master_node = ddp_rank == 0
    seed_offset = ddp_rank  # Each process gets a different seed
    assert grad_accumulation_steps % ddp_world_size == 0
    grad_accumulation_steps //= ddp_world_size
else:
    # Single GPU training setup
    is_master_node = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iteration = grad_accumulation_steps * ddp_world_size * micro_batch_size * context_length
print(f"Tokens per iteration: {tokens_per_iteration:,}")

# Output directory setup
if is_master_node:
    os.makedirs(output_directory, exist_ok=True)

# Seed and precision setup
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in selected_device else 'cpu'
precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[tensor_dtype]
execution_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=precision_dtype)

# Data loading
data_directory = os.path.join('data', dataset_name)
def load_batch(split):
    if split == 'train':
        data_file = np.memmap(os.path.join(data_directory, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data_file = np.memmap(os.path.join(data_directory, 'val.bin'), dtype=np.uint16, mode='r')
    idx = torch.randint(len(data_file) - context_length, (micro_batch_size,))
    x = torch.stack([torch.from_numpy(data_file[i:i + context_length].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data_file[i + 1:i + 1 + context_length].astype(np.int64)) for i in idx])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(selected_device, non_blocking=True), y.pin_memory().to(selected_device, non_blocking=True)
    else:
        x, y = x.to(selected_device), y.to(selected_device)
    return x, y

# Training state initialization
iteration_count = 0
best_validation_loss = float('inf')

# Attempt to derive vocab_size from the dataset metadata
meta_file = os.path.join(data_directory, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_file):
    with open(meta_file, 'rb') as f:
        metadata = pickle.load(f)
    meta_vocab_size = metadata['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size} in {meta_file}")

# Model initialization
model_parameters = dict(n_layer=num_layers, n_head=num_attention_heads, n_embd=embedding_size, block_size=context_length,
                        bias=use_bias_in_model, vocab_size=None, dropout=dropout_rate)

if initialization_mode == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("Defaulting to GPT-2 vocab_size: 50304")
    model_parameters['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
    gpt_configuration = GPTConfig(**model_parameters)
    model_instance = GPT(gpt_configuration)

elif initialization_mode == 'resume':
    print(f"Resuming training from {output_directory}")
    checkpoint_path = os.path.join(output_directory, 'checkpoint.pt')
    checkpoint_data = torch.load(checkpoint_path, map_location=selected_device)
    checkpoint_model_params = checkpoint_data['model_args']
    for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_parameters[key] = checkpoint_model_params[key]
    gpt_configuration = GPTConfig(**model_parameters)
    model_instance = GPT(gpt_configuration)
    model_instance.load_state_dict(checkpoint_data['model'])
    iteration_count = checkpoint_data['iter_num']
    best_validation_loss = checkpoint_data['best_val_loss']

elif initialization_mode.startswith('gpt2'):
    print(f"Initializing from GPT-2 weights: {initialization_mode}")
    model_instance = GPT.from_pretrained(initialization_mode, dict(dropout=dropout_rate))
    for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_parameters[key] = getattr(model_instance.config, key)

if context_length < model_instance.config.block_size:
    model_instance.crop_block_size(context_length)
    model_parameters['block_size'] = context_length

model_instance.to(selected_device)

# Gradient scaling for mixed precision
grad_scaler = torch.cuda.amp.GradScaler(enabled=(tensor_dtype == 'float16'))

# Optimizer initialization
optimizer = model_instance.configure_optimizers(weight_decay_rate, base_learning_rate, (beta1, beta2), device_type)
if initialization_mode == 'resume':
    optimizer.load_state_dict(checkpoint_data['optimizer'])
checkpoint_data = None  # Free memory

# Compile model if enabled
if enable_compilation:
    print("Compiling model (this may take a few minutes)...")
    uncompiled_model = model_instance
    model_instance = torch.compile(model_instance)

# Wrap model for DDP if needed
if distributed_training:
    model_instance = DDP(model_instance, device_ids=[ddp_local_rank])

# Helper function to estimate loss
@torch.no_grad()
def evaluate_loss():
    results = {}
    model_instance.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_steps)
        for i in range(evaluation_steps):
            input_batch, target_batch = load_batch(split)
            with execution_context:
                predictions, batch_loss = model_instance(input_batch, target_batch)
            losses[i] = batch_loss.item()
        results[split] = losses.mean()
    model_instance.train()
    return results

# Learning rate scheduler
def adjust_learning_rate(iteration):
    if iteration < warmup_iterations:
        return base_learning_rate * iteration / warmup_iterations
    if iteration > lr_decay_total_steps:
        return minimum_learning_rate
    decay_ratio = (iteration - warmup_iterations) / (lr_decay_total_steps - warmup_iterations)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return minimum_learning_rate + cosine_decay * (base_learning_rate - minimum_learning_rate)

# Logging setup
if wandb_enabled and is_master_node:
    import wandb
    wandb.init(project=wandb_project_name, name=wandb_run_identifier, config=config)

# Training loop
first_batch_inputs, first_batch_targets = load_batch('train')
start_time = time.time()
local_iteration_count = 0
unwrapped_model = model_instance.module if distributed_training else model_instance
mean_flops_utilization = -1.0

while True:
    # Adjust learning rate for this iteration
    lr = adjust_learning_rate(iteration_count) if enable_lr_decay else base_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation and checkpointing
    if iteration_count % evaluation_interval == 0 and is_master_node:
        evaluation_losses = evaluate_loss()
        print(f"Iteration {iteration_count}: train loss {evaluation_losses['train']:.4f}, val loss {evaluation_losses['val']:.4f}")
        if wandb_enabled:
            wandb.log({
                "iteration": iteration_count,
                "train/loss": evaluation_losses['train'],
                "val/loss": evaluation_losses['val'],
                "learning_rate": lr,
                "mfu": mean_flops_utilization * 100,  # Convert to percentage
            })
        if evaluation_losses['val'] < best_validation_loss or save_checkpoint_every_eval:
            best_validation_loss = evaluation_losses['val']
            if iteration_count > 0:
                checkpoint = {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_parameters,
                    'iter_num': iteration_count,
                    'best_val_loss': best_validation_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {output_directory}")
                torch.save(checkpoint, os.path.join(output_directory, 'checkpoint.pt'))
    if iteration_count == 0 and evaluation_mode:
        break

    # Forward-backward pass with gradient accumulation
    for step in range(grad_accumulation_steps):
        if distributed_training:
            model_instance.require_backward_grad_sync = (step == grad_accumulation_steps - 1)
        with execution_context:
            predictions, loss = model_instance(first_batch_inputs, first_batch_targets)
            loss = loss / grad_accumulation_steps
        first_batch_inputs, first_batch_targets = load_batch('train')
        grad_scaler.scale(loss).backward()

    # Gradient clipping
    if gradient_clip_value != 0.0:
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), gradient_clip_value)

    # Optimizer step and gradient scaler update
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    end_time = time.time()
    time_elapsed = end_time - start_time
    start_time = end_time
    if iteration_count % logging_interval == 0 and is_master_node:
        loss_value = loss.item() * grad_accumulation_steps
        if local_iteration_count >= 5:  # Allow loop to settle before calculating MFU
            flops_utilization = unwrapped_model.estimate_mfu(micro_batch_size * grad_accumulation_steps, time_elapsed)
            mean_flops_utilization = flops_utilization if mean_flops_utilization == -1.0 else 0.9 * mean_flops_utilization + 0.1 * flops_utilization
        print(f"Iteration {iteration_count}: loss {loss_value:.4f}, time {time_elapsed * 1000:.2f}ms, MFU {mean_flops_utilization * 100:.2f}%")

    iteration_count += 1
    local_iteration_count += 1

    # Check for termination
    if iteration_count > total_training_iterations:
        break

# Cleanup for DDP
if distributed_training:
    destroy_process_group()
