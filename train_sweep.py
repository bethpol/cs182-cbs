"""
Training script for hyperparameter sweep on AdamW and Muon with CBS DataLoader.

Features:
- Token-based checkpointing (absolute_token_pos)
- WandB logging (train_loss, val_loss, grad_norm)
- Uses 45M GPT model
- Single GPU training only

Features removed from nanoGPT train script:
- DDP training
- Gradient scaling
- learning rate scheduler

Usage:
    Single GPU:
        python train_cbs.py --batch_size=32

Customization:
    You can easily switch optimizers, adjust hyperparameters, or add
    learning rate schedulers without modifying the model code.
"""

import os
import sys
import time
import math
import inspect
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model_45M import build_gpt_45m, count_parameters
from dataloader import create_dataloader
from optimizers import adamw, muon_w_adam

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out_cbs'
init_from = 'scratch'              # 'scratch' or 'resume'
resume_ckpt_fname = ""
log_interval_tokens = 5120      # Log every 100K tokens
checkpoint_interval_tokens = 5_000_000  # Checkpoint every 5M tokens
eval_iters = 200                   # Number of evaluation iterations (steps) not on a token basis

# WandB logging
wandb_log = True
wandb_project = 'hyperparam-search'
wandb_run_name = out_dir

# Data
train_data_file = 'data/c4_dataset/100M/train_shuffled_1024.bin'
val_data_file = 'data/c4_dataset/100M/val.bin'
block_size = 1024
total_tokens = None  # Auto-detect from file
checkpoint_token_pos = 0  # Starting position (for resume)
branch_seed = -1  # No branching by default
branch_window_size_tokens = 100000000  # Required parameter

# Training
batch_size = 32  # Per-GPU batch size
gradient_accumulation_steps = 1
max_tokens = 1_000_000_000  # Train for 1B tokens

# AdamW optimizer (no scheduler)
optimizer_type = "adamw" # str: adamw or muon
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = None  # Gradient clipping
lr_muon_adam = 3e-4
wd_muon_adam = 0.1

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = True  # torch.compile (set to True for PyTorch 2.0+)

# DDP
backend = 'nccl'

# -----------------------------------------------------------------------------
# Parse command line arguments
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# -----------------------------------------------------------------------------
# No DDP for now
# -----------------------------------------------------------------------------
master_process = True
train_seed = 0
ddp_world_size = 1
ddp_rank = 0
ddp_local_rank = 0
print("[Single GPU mode]")

# Create output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Set seed
torch.manual_seed(train_seed)
np.random.seed(train_seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Setup autocast context
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Initialize model
# -----------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Initializing model...")
print(f"{'='*60}")

if init_from == 'scratch':
    print("Creating new 45M GPT model from scratch")
    model = build_gpt_45m(device=device)
    checkpoint_token_pos = 0
    tokens_seen = 0
    optimizer_state = None
    best_val_loss = float('inf')
elif init_from == 'resume':
    print(f"Resuming from checkpoint in {out_dir}")
    if resume_ckpt_fname == "":
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, resume_ckpt_fname)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_gpt_45m(device=device)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    checkpoint_token_pos = checkpoint['checkpoint_token_pos'] # dataloader_file_position
    tokens_seen = checkpoint['tokens_seen'] # training_tokens_processed or total_tokens_trained in a certain/ this run
    optimizer_state = checkpoint['optimizer']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Resumed from token position: {checkpoint_token_pos:,}")
    print(f"Tokens seen so far: {tokens_seen:,}")
else:
    raise ValueError(f"Unknown init_from: {init_from}")

print(f"Model parameters: {count_parameters(model):,}")

# -----------------------------------------------------------------------------
# Initialize optimizer
# -----------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Initializing optimizer...")
print(f"{'='*60}")

# Create optimizer directly in training script for flexibility
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if optimizer_type == "adamw":
    optimizer = adamw(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
    )

    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Betas: ({beta1}, {beta2})")
elif optimizer_type == "muon":
    optimizer = muon_w_adam(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_adam=lr_muon_adam,
        wd_adam=wd_muon_adam,
        beta1=beta1,
        beta2=beta2
    )
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
else:
    raise NotImplementedError(f"Code not implemented to train with: {optimizer_type} optimizer")

if init_from == 'resume' and optimizer_state is not None:
    optimizer.load_state_dict(optimizer_state)
    print("Loaded optimizer state from checkpoint")

# Compile model if requested
if compile_model:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Initialize dataloaders
# -----------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Initializing dataloaders...")
print(f"{'='*60}")

train_loader = create_dataloader(
    data_file=train_data_file,
    block_size=block_size,
    checkpoint_token_pos=checkpoint_token_pos,
    branch_seed=branch_seed,
    branch_window_size_tokens=branch_window_size_tokens,
    device=device,
)

# Load dataloader state if resuming
if init_from == 'resume':
    train_loader.load_state(checkpoint['dataloader_state'])
    print(f"Loaded dataloader state")

# Validation loader (always starts from beginning)
val_loader = create_dataloader(
    data_file=val_data_file,
    block_size=block_size,
    checkpoint_token_pos=0,
    branch_seed=-1,
    branch_window_size_tokens=branch_window_size_tokens,
    device=device,
)

# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters: int) -> float:
    """Estimate validation loss."""
    model.eval()
    losses = []

    # Reset validation loader to start
    val_loader.current_sequence_idx = 0

    for _ in range(eval_iters):
        try:
            x, y = val_loader.get_batch(batch_size) # same batch size as the training batch size
            with ctx:
                logits, loss = model(x, y)
            losses.append(loss.item())
        except StopIteration:
            # Reset if we run out of validation data
            val_loader.current_sequence_idx = 0
            break

    model.train()

    if len(losses) == 0:
        return float('inf')

    return np.mean(losses)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_state: Dict,
    checkpoint_token_pos: int,
    tokens_seen: int,
    global_iter: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    out_dir: str,
    ckpt_name: str,
    is_best: bool = False
):
    """Save checkpoint with token position."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dataloader_state': dataloader_state,
        'checkpoint_token_pos': checkpoint_token_pos,
        'tokens_seen': tokens_seen,
        'global_iter': global_iter,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config
    }

    # Save regular checkpoint
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    # print(f"Saved checkpoint to {ckpt_path}")

    # Save best checkpoint if this is the best
    if is_best:
        best_path = os.path.join(out_dir, f'best.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved BEST checkpoint to {best_path}")


# -----------------------------------------------------------------------------
# Initialize WandB
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb

    # config_wandb = {
    #     'model': '45M-GPT',
    #     'batch_size': batch_size * gradient_accumulation_steps * ddp_world_size,
    #     'learning_rate': learning_rate,
    #     'weight_decay': weight_decay,
    #     'block_size': block_size,
    #     'max_tokens': max_tokens,
    #     'grad_clip': grad_clip,
    #     'dtype': dtype,
    #     'num_gpus': ddp_world_size,
    # }

    wandb.init(
        entity="cs182-cbs",
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        # resume='allow' if init_from == 'resume' else None
    )
    print(f"WandB initialized: {wandb_project}/{wandb_run_name}")

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Starting training...")
print(f"{'='*60}")

model.train()
t0 = time.time()
local_iter = 0
global_iter = 0 # global iteration during a training run not during logging intervals. not sure if i want to grab this from the checkpoint as well for branched training
train_loss = 0.0
last_log_tokens = tokens_seen
last_checkpoint_tokens = tokens_seen

# Calculate tokens per iteration
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * ddp_world_size

print(f"\nTraining configuration:")
print(f"  Tokens per iteration: {tokens_per_iter:,}")
print(f"  Starting tokens seen: {tokens_seen:,}")
print(f"  Max tokens: {max_tokens:,}")
print(f"  Log interval: {log_interval_tokens:,} tokens")
print(f"  Checkpoint interval: {checkpoint_interval_tokens:,} tokens")
print()

while tokens_seen < max_tokens:
    # Forward-backward pass with gradient accumulation
    loss_accum = 0.0

    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = train_loader.get_batch(batch_size)
        except StopIteration:
            print(f"\nReached end of training data at {tokens_seen:,} tokens")
            break

        # Forward pass
        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()
        loss_accum += loss.item()
        
    # calculate grad_norm with or without clipping
    scaler.unscale_(optimizer)
    if grad_clip != None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Update tokens seen
    tokens_seen += tokens_per_iter
    train_loss += loss_accum
    local_iter += 1
    global_iter += 1

    # Logging
    avg_loss = train_loss / local_iter
    if tokens_seen - last_log_tokens >= log_interval_tokens and master_process:
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (tokens_seen - last_log_tokens) / dt

        print(f"tokens: {tokens_seen:>12,} | loss: {avg_loss:.4f} | "
                f"grad_norm: {grad_norm:.4f} | tok/s: {tokens_per_sec:>8,.0f} | "
                f"time: {dt:.2f}s")
        
        # Evaluation
        val_loss = estimate_loss(model, val_loader, eval_iters)
        print(f"Validation loss: {val_loss:.4f}")

        if wandb_log:
            wandb.log({
                'global_iter': global_iter,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'grad_norm': grad_norm,
                'train_tokens_seen': tokens_seen,
            }, step=tokens_seen)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")

        last_log_tokens = tokens_seen
        # reset loss calculations after each log
        train_loss = 0.0
        local_iter = 0
        t0 = time.time()

    # Checkpointing
    if tokens_seen - last_checkpoint_tokens >= checkpoint_interval_tokens and master_process and tokens_seen>0:
        print(f"\nSaving checkpoint at {tokens_seen:,} tokens...")
        if log_interval_tokens != checkpoint_interval_tokens:
            val_loss = estimate_loss(model, val_loader, eval_iters)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
        dataloader_state = train_loader.get_state()
        tokens_str = f"{tokens_seen:.2e}".replace("+", "").replace("e0", "e")  # e.g., 1.23e8
        train_loss_str = f"{avg_loss:.3f}"
        val_loss_str = f"{val_loss:.3f}"
        # ckpt_name = f"tokens({tokens_str})_tloss({train_loss_str})_vloss({val_loss_str})_ckpt.pt"
        ckpt_name = "checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            dataloader_state=dataloader_state,
            checkpoint_token_pos=dataloader_state['token_pos'],
            tokens_seen=tokens_seen,
            global_iter=global_iter,
            train_loss=avg_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            out_dir=out_dir,
            ckpt_name=ckpt_name,
            is_best=is_best # not really used as effectively yet
        )

        last_checkpoint_tokens = tokens_seen

# Final checkpoint and evaluation
if master_process:
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Final tokens seen: {tokens_seen:,}")

    # Final evaluation
    print("Running final evaluation...")
    val_loss = estimate_loss(model, val_loader, eval_iters)
    print(f"Final validation loss: {val_loss:.4f}")

    # Save final checkpoint
    print("Saving final checkpoint...")
    dataloader_state = train_loader.get_state()
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        print(f"Best validation loss: {best_val_loss:.4f}")
    tokens_str = f"{tokens_seen:.2e}".replace("+", "").replace("e0", "e")  # e.g., 1.23e8
    train_loss_str = f"{avg_loss:.3f}"
    val_loss_str = f"{val_loss:.3f}"
    # ckpt_name = f"tokens({tokens_str})_tloss({train_loss_str})_vloss({val_loss_str})_ckpt.pt"
    ckpt_name = "checkpoint_last.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        dataloader_state=dataloader_state,
        checkpoint_token_pos=dataloader_state['token_pos'],
        tokens_seen=tokens_seen,
        global_iter=global_iter,
        train_loss=avg_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        out_dir=out_dir,
        ckpt_name=ckpt_name,
        is_best=is_best
    )

if wandb_log and master_process:
    wandb.finish()

print("\nDone Training!")