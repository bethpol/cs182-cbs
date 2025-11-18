"""
Training script for hyperparameter sweep on AdamW and Muon with CBS DataLoader.

Features:
- Token-based checkpointing (absolute_token_pos)
- WandB logging (train_loss, val_loss, grad_norm)
- Uses 45M GPT model
- Single GPU training only
- Gradient scaling
- learning rate scheduler

Features removed from nanoGPT train script:
- DDP training

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
import glob
import inspect
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from model_45M import build_gpt_45m, count_parameters
from dataloader import create_dataloader
from optimizers import adamw, muon_w_adam
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# I/O
out_dir = '/data/ejweaver/out_cbs/adamw_long'
init_from = 'scratch'              # 'scratch' or 'resume'
resume_ckpt_fname = ""
log_interval_tokens = 5_000_000      # Log training loss every 5M tokens
val_interval_tokens = 50_000_000     # Evaluate validation loss every 50M tokens (less frequent)
checkpoint_interval_tokens = 500_000_000  # Checkpoint every 500M tokens
eval_iters = 200                   # Number of evaluation iterations (steps) not on a token basis
eval_only = False

# WandB logging
wandb_log = True
wandb_project = 'cbs-train'
wandb_run_name = out_dir

# Data
train_data_file = 'data/c4_dataset/100M/train_shuffled_512.bin'
val_data_file = 'data/c4_dataset/100M/val.bin'
block_size = 512
total_tokens = None  # Auto-detect from file
checkpoint_token_pos = 0  # Starting position (for resume)
branch_seed = -1  # No branching by default
branch_window_size_tokens = 100000000  # Required parameter

# Training
batch_size = 32  # Per-GPU batch size
gradient_accumulation_steps = 1
max_tokens = 5_000_000_000  # Train for 1B tokens

# AdamW or Muon optimizer
optimizer_type = "adamw" # str: adamw or muon
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # Gradient clipping
lr_muon_adam = 3e-4
wd_muon_adam = 0.1

# Learning rate scheduler (unified 2-phase or 3-phase)
use_scheduler = True           # Set to True to enable
warmup_tokens = 50_000_000      # Warmup phase
stable_tokens = 0     # Stable phase for a 3phase setup. set to 0 for 2-phase i.e. if you don't want a stable phase and want to go to the decay phase directly after warmup
min_lr_factor = 0.1                  # Minimum LR factor in range [0,1]

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
# DDP setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1  # Is this a DDP run?
if ddp:
    # Initialize DDP - must happen BEFORE setting device
    init_process_group(backend=backend)
    
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    master_process = ddp_rank == 0
    train_seed = 0
    print(f"[DDP] Rank {ddp_rank}/{ddp_world_size}, Local rank: {ddp_local_rank}, Device: {device}")
else:
    # Single GPU mode
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
    # crop down the model block size if desired, using model surgery -- if you want to train with smaller number of tokens per sequence
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
    print(f"Model block size: {model.config.block_size}")
    checkpoint_token_pos = 0
    tokens_seen = 0
    global_iter_resume = 0
    optimizer_state = None
    best_val_loss = float('inf')
elif init_from == 'resume':
    print(f"Resuming from checkpoint in {out_dir}")
    if resume_ckpt_fname == "":
        # Try checkpoint_last.pt first
        ckpt_path = os.path.join(out_dir, 'checkpoint_last.pt')
        if not os.path.exists(ckpt_path):
            # Find the most recent milestone checkpoint (checkpoint_*M.pt)
            milestone_ckpts = glob.glob(os.path.join(out_dir, 'checkpoint_*M.pt'))
            if milestone_ckpts:
                # Extract numbers and find the largest
                def extract_milestone(path):
                    basename = os.path.basename(path)  # e.g., "checkpoint_1500M.pt"
                    try:
                        num_str = basename.split('_')[1].replace('M.pt', '')  # "1500"
                        return int(num_str)
                    except:
                        return -1
                
                milestone_ckpts.sort(key=extract_milestone, reverse=True)
                ckpt_path = milestone_ckpts[0]
                print(f"checkpoint_last.pt not found, using most recent milestone: {os.path.basename(ckpt_path)}")
            else:
                # Fall back to ckpt.pt for backwards compatibility
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    else:
        # If resume_ckpt_fname is an absolute path, use it directly
        # Otherwise, join with out_dir (for relative paths)
        if os.path.isabs(resume_ckpt_fname):
            ckpt_path = resume_ckpt_fname
        else:
            ckpt_path = os.path.join(out_dir, resume_ckpt_fname)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_gpt_45m(device=device)
    # crop down the model block size if desired, using model surgery -- if you want to train with smaller number of tokens per sequence
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
    print(f"Model block size: {model.config.block_size}")
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
    global_iter_resume = checkpoint.get('global_iter', 0)  # Restore global iteration count
    optimizer_state = checkpoint['optimizer']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # For branching experiments, reset tokens_seen to 0 so each branch trains for the same duration
    if branch_seed >= 0:
        print(f"\n{'='*60}")
        print(f"BRANCHING EXPERIMENT DETECTED")
        print(f"{'='*60}")
        print(f"  Source checkpoint: {ckpt_path}")
        print(f"  Checkpoint's token position: {checkpoint_token_pos:,}")
        print(f"  Checkpoint's tokens seen: {tokens_seen:,}")
        print(f"  Branch seed: {branch_seed}")
        print(f"  → RESETTING tokens_seen to 0 for this branch")
        print(f"  → RESETTING global_iter to 0 for this branch")
        tokens_seen = 0
        global_iter_resume = 0
        print(f"  Branch will train from tokens_seen=0")
        print(f"{'='*60}")
    else:
        print(f"Resumed from checkpoint: {ckpt_path}")
        print(f"  Token position: {checkpoint_token_pos:,}")
        print(f"  Tokens seen: {tokens_seen:,}")
        print(f"  Global iterations: {global_iter_resume:,}")
        print(f"  Best val loss: {best_val_loss:.4f}")
else:
    raise ValueError(f"Unknown init_from: {init_from}")

print(f"Model parameters: {count_parameters(model):,}")
print(f"Model on device: {next(model.parameters()).device}")

# -----------------------------------------------------------------------------
# Compile and wrap model BEFORE creating optimizer
# -----------------------------------------------------------------------------
# Compile model if requested (do this BEFORE DDP wrapping)
if compile_model:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

# Wrap model in DDP if needed (BEFORE optimizer creation)
if ddp:
    print(f"[DDP] Rank {ddp_rank}: Wrapping model in DDP...")
    model = DDP(
        model, 
        device_ids=[ddp_local_rank], 
        output_device=ddp_local_rank,
        find_unused_parameters=False,
        broadcast_buffers=True,
    )
    print(f"[DDP] Rank {ddp_rank}: Model wrapped successfully")

# -----------------------------------------------------------------------------
# Initialize optimizer (AFTER DDP wrapping)
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

if init_from == 'resume' and optimizer_state is not None and branch_seed < 0:
    # For normal resume: load optimizer state
    # For branching: start with fresh optimizer state (new trajectory)
    optimizer.load_state_dict(optimizer_state)
    print("Loaded optimizer state from checkpoint")

# Create learning rate scheduler
scheduler = None
if use_scheduler:
    tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * ddp_world_size
    max_steps = max_tokens // tokens_per_iter
    warmup_steps = warmup_tokens // tokens_per_iter
    stable_steps = stable_tokens // tokens_per_iter

    # Adjust stable_tokens based on scheduler_type
    if stable_steps > 0:
        print(f"Using 3-phase scheduler (warmup → stable → decay)")
    else:
        print("Using 2-phase scheduler (warmup → decay)")

    decay_start_step = warmup_steps + stable_steps

    def lr_lambda(current_step):
        """
        Unified lambda function for both 2-phase and 3-phase schedulers.
        - 2-phase: warmup → cosine decay (stable_steps = 0)
        - 3-phase: warmup → stable → cosine decay (stable_steps > 0)
        """
        # Phase 1: Warmup (0 → 1.0)
        if current_step < warmup_steps:
            return (current_step+1) / max(1, warmup_steps) # 0-indexed step, hence + 1 adjustments

        # Phase 2: Stable at peak (only active if stable_steps > 0)
        if current_step < decay_start_step:
            return 1.0

        # Phase 3: Cosine decay (1.0 → min_lr_factor) min_lr_factor = min_lr/learning_rate
        progress = (current_step - decay_start_step) / max(1, (max_steps - decay_start_step))
        progress = min(1.0, progress)  # Clamp to 1.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Load scheduler state if resuming (but not for branching - branches start fresh)
    if init_from == 'resume' and 'scheduler' in checkpoint and branch_seed < 0:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"\nLoaded scheduler state from checkpoint")
        print(f"  Resuming from step: {scheduler.last_epoch}") # last_epoch: Current step count (despite the confusing name)
        print(f"  Current learning rate: {scheduler.get_last_lr()[0]:.6e}")

    print(f"  Warmup: {warmup_steps} steps ({warmup_tokens:,} tokens)")
    if stable_steps > 0:
        print(f"  Stable: {stable_steps} steps ({stable_tokens:,} tokens)")
    print(f"  Decay: {max_steps - decay_start_step} steps")
    print(f"  Total: {max_steps} steps ({max_tokens:,} tokens)")
    print(f"  Peak LR for optimizer of interest: {learning_rate}, Min LR for optimizer of interest: {min_lr_factor*learning_rate}")

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
    num_gpus=ddp_world_size,
    gpu_rank=ddp_rank,
)

# Load dataloader state if resuming (but NOT for branching experiments)
# For branching (branch_seed >= 0), we want to start from a NEW data position,
# not continue from the checkpoint's old position
if init_from == 'resume' and 'dataloader_state' in checkpoint and branch_seed < 0:
    train_loader.load_state(checkpoint['dataloader_state'])
    print(f"Loaded dataloader state - resuming from sequence position")
elif branch_seed >= 0:
    print(f"Branching experiment (seed={branch_seed}) - starting from fresh data position")
    print(f"  Checkpoint's saved position: {checkpoint_token_pos:,}")
    print(f"  Offset (seed × window): {branch_seed} × {branch_window_size_tokens:,} = {branch_seed * branch_window_size_tokens:,}")
    print(f"  Final start position: {train_loader.start_token_pos:,} (checkpoint + offset)")

# Validation loader (always starts from beginning)
val_loader = create_dataloader(
    data_file=val_data_file,
    block_size=block_size,
    checkpoint_token_pos=0,
    branch_seed=-1,
    branch_window_size_tokens=branch_window_size_tokens,
    device=device,
    num_gpus=ddp_world_size,
    gpu_rank=ddp_rank,
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

    local_loss = np.mean(losses)
    
    # Average loss across all GPUs in DDP
    if ddp:
        loss_tensor = torch.tensor(local_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()
    
    return local_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
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

    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    # Save regular checkpoint
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    # print(f"Saved checkpoint to {ckpt_path}")

    # Save best checkpoint if this is the best
    if is_best:
        best_path = os.path.join(out_dir, f'best_vloss_{val_loss:.4f}.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved BEST checkpoint to {best_path} with validation loss {val_loss:.4f}")


# -----------------------------------------------------------------------------
# Initialize WandB
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb

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
global_iter = global_iter_resume  # Resume from checkpoint or start at 0
train_loss = 0.0
val_loss = float('inf')  # Initialize for checkpointing
# Track milestones to avoid drift (saves at 5M, 10M, 15M... not 5M, 10.01M, 15.02M...)
log_milestone_count = tokens_seen // log_interval_tokens
val_milestone_count = tokens_seen // val_interval_tokens
checkpoint_milestone_count = tokens_seen // checkpoint_interval_tokens
last_log_tokens = tokens_seen  # For throughput calculation

# Calculate tokens per iteration
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * ddp_world_size

print(f"\nTraining configuration:")
print(f"  Tokens per iteration: {tokens_per_iter:,}")
print(f"  Starting tokens seen: {tokens_seen:,}")
print(f"  Max tokens: {max_tokens:,}")
print(f"  Train log interval: {log_interval_tokens:,} tokens")
print(f"  Val log interval: {val_interval_tokens:,} tokens")
print(f"  Checkpoint interval: {checkpoint_interval_tokens:,} tokens")

if init_from == 'resume':
    next_log_milestone = (log_milestone_count + 1) * log_interval_tokens
    next_val_milestone = (val_milestone_count + 1) * val_interval_tokens
    next_ckpt_milestone = (checkpoint_milestone_count + 1) * checkpoint_interval_tokens
    
    if branch_seed >= 0:
        print(f"\nBranching milestones (starting fresh from 0):")
    else:
        print(f"\nResuming from milestones:")
    
    print(f"  Last train log milestone: {log_milestone_count * log_interval_tokens:,} → next at {next_log_milestone:,}")
    print(f"  Last val log milestone: {val_milestone_count * val_interval_tokens:,} → next at {next_val_milestone:,}")
    print(f"  Last checkpoint milestone: {checkpoint_milestone_count * checkpoint_interval_tokens:,} → next at {next_ckpt_milestone:,}")

print()

data_loader_exhausted = False

while tokens_seen < max_tokens:
    # Forward-backward pass with gradient accumulation
    loss_accum = 0.0

    if eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = train_loader.get_batch(batch_size)
        except StopIteration:
            print(f"\nReached end of training data at {tokens_seen:,} tokens")
            data_loader_exhausted = True
            break

        # Forward pass
        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()
        loss_accum += loss.item()

    if data_loader_exhausted:
        break
        
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

    # Step the scheduler
    if use_scheduler:
        scheduler.step()

    # Update tokens seen
    tokens_seen += tokens_per_iter
    train_loss += loss_accum
    local_iter += 1
    global_iter += 1

    # Training loss logging (frequent) - milestone-based to avoid drift
    avg_loss = train_loss / local_iter
    current_log_milestone = tokens_seen // log_interval_tokens
    if current_log_milestone > log_milestone_count:
        log_milestone_count = current_log_milestone
        
        # Only master process logs and prints training metrics
        if master_process:
            t1 = time.time()
            dt = t1 - t0
            tokens_since_last_log = tokens_seen - last_log_tokens
            tokens_per_sec = tokens_since_last_log / dt

            print(f"tokens: {tokens_seen:>12,} | loss: {avg_loss:.4f} | "
                    f"grad_norm: {grad_norm:.4f} | tok/s: {tokens_per_sec:>8,.0f} | "
                    f"time: {dt:.2f}s")

            if wandb_log:
                wandb.log({
                    'global_iter': global_iter,
                    'train_loss': avg_loss,
                    'grad_norm': grad_norm,
                    'train_tokens_seen': tokens_seen,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, step=tokens_seen)

        # reset loss calculations after each log
        last_log_tokens = tokens_seen
        train_loss = 0.0
        local_iter = 0
        t0 = time.time()
    
    # Validation loss logging (less frequent) - milestone-based to avoid drift
    current_val_milestone = tokens_seen // val_interval_tokens
    if current_val_milestone > val_milestone_count:
        val_milestone_count = current_val_milestone
        
        # Evaluation (must be called by ALL processes for DDP synchronization)
        val_loss = estimate_loss(model, val_loader, eval_iters)
        
        # Only master process logs and prints
        if master_process:
            print(f"Validation loss: {val_loss:.4f}")

            if wandb_log:
                wandb.log({
                    'val_loss': val_loss,
                }, step=tokens_seen)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")

    # Checkpointing - save at each milestone (500M, 1000M, 1500M, etc.)
    current_ckpt_milestone = tokens_seen // checkpoint_interval_tokens
    if current_ckpt_milestone > checkpoint_milestone_count and tokens_seen > 0:
        # We've crossed a new checkpoint milestone!
        checkpoint_milestone_count = current_ckpt_milestone
        
        # Evaluate if we haven't just computed validation loss (must be called by ALL processes)
        # Check if val was just computed by comparing milestones
        if current_val_milestone == val_milestone_count:
            # We just computed val_loss in the validation logging block above
            pass
        else:
            # Need to compute val_loss for this checkpoint
            val_loss = estimate_loss(model, val_loader, eval_iters)
        
        # Only master process saves checkpoints
        if master_process:
            milestone_tokens = current_ckpt_milestone * checkpoint_interval_tokens
            print(f"\nSaving checkpoint at {tokens_seen:,} tokens (milestone: {milestone_tokens:,})...")
            
            # Check if this is the best model so far
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
            
            dataloader_state = train_loader.get_state()
            # Use the milestone for the checkpoint name (e.g., checkpoint_500M.pt, checkpoint_1000M.pt)
            milestone_millions = milestone_tokens // 1_000_000
            ckpt_name = f"checkpoint_{milestone_millions}M.pt"
            model_to_save = model.module if ddp else model
            save_checkpoint(
                model=model_to_save,
                optimizer=optimizer,
                scheduler=scheduler if use_scheduler else None,
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

# Final checkpoint and evaluation
# Final evaluation (must be called by ALL processes for DDP synchronization)
val_loss = estimate_loss(model, val_loader, eval_iters)

if master_process:
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Final tokens seen: {tokens_seen:,}")
    print(f"Final validation loss: {val_loss:.4f}")

    # Save final checkpoint
    print("Saving final checkpoint...")
    dataloader_state = train_loader.get_state()
    is_best = val_loss < best_val_loss
    print(f"Validation loss: {val_loss:.4f}")
    if is_best:
        best_val_loss = val_loss
        print(f"Best validation loss: {best_val_loss:.4f}")
    # tokens_str = f"{tokens_seen:.2e}".replace("+", "").replace("e0", "e")  # e.g., 1.23e8
    # Calculate final avg_loss for checkpoint
    avg_loss = train_loss / local_iter if local_iter > 0 else 0.0
    # train_loss_str = f"{avg_loss:.3f}"
    # val_loss_str = f"{val_loss:.3f}"
    # # ckpt_name = f"tokens({tokens_str})_tloss({train_loss_str})_vloss({val_loss_str})_ckpt.pt"
    ckpt_name = "checkpoint_last.pt"
    model_to_save = model.module if ddp else model
    save_checkpoint(
        model=model_to_save,
        optimizer=optimizer,
        scheduler=scheduler if use_scheduler else None,
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

# Clean up DDP
if ddp:
    destroy_process_group()

print("\nDone Training!")