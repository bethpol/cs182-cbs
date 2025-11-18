"""
Generate configuration files for branched training experiments across MULTIPLE CHECKPOINTS
with MULTIPLE SEEDS per checkpoint.

This script extends the basic branching experiment to:
1. Branch from multiple checkpoints (e.g., checkpoints at different training stages)
2. For each checkpoint, create branches with different random seeds
3. Automatically handle batch sizes with gradient accumulation
4. Support multi-GPU training

Use case: Testing how branching behavior varies across different checkpoints and seeds
"""

import os
import math
import re
import auto_gradient_accumulation as aga

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE VALUES
# =============================================================================

# Multiple checkpoints to branch from
# Format: List of (checkpoint_dir, checkpoint_file, checkpoint_step) tuples
# checkpoint_step is used for naming/tracking (e.g., "ckpt_100k" for checkpoint at 100k steps)
CHECKPOINTS = [
    ("/data/ejweaver/out_cbs/adamw_full_5B_run_fixed", "checkpoint_500M.pt", "500M"),
    ("/data/ejweaver/out_cbs/adamw_full_5B_run_fixed", "checkpoint_2500M.pt", "1000M"),
    ("/data/ejweaver/out_cbs/adamw_full_5B_run_fixed", "checkpoint_3500M.pt", "1500M"),
    ("/data/ejweaver/out_cbs/adamw_full_5B_run_fixed", "checkpoint_4500M.pt", "2000M"),
]

# Multiple seeds to test per checkpoint
BRANCH_SEEDS = [0, 1, 2]

# Base hyperparameters (from your original training run)
BASE_BATCH_SIZE = 32  # For CBS experiments
BASE_LEARNING_RATE = 3e-4
BASE_BLOCK_SIZE = 512

# Experiment parameters
K_VALUES = [1, 2, 4, 8, 16, 32]
DELTA_STEPS_AS_TOKENS = 100_663_296  # Window size as requested
BRANCH_WINDOW_SIZE_TOKENS = 100_663_296  # Same as above

# Data paths
TRAIN_DATA_FILE = '/data/ejweaver/c4_subset/train_shuffled_512.bin'
VAL_DATA_FILE = '/data/ejweaver/c4_subset/val_large.bin'

# Optimizer settings
OPTIMIZER_TYPE = "adamw"
WEIGHT_DECAY = 0.003
LEARNING_RATE_MUON_ADAM = 3e-4
WEIGHT_DECAY_MUON_ADAM = 0.003
BETA1 = 0.9
BETA2 = 0.95

# Logging and checkpointing
LOG_INTERVAL_TOKENS = 10_000
CHECKPOINT_INTERVAL_TOKENS = 50_000
EVAL_ITERS = 100

# WandB settings
WANDB_LOG = True
WANDB_PROJECT = "branched-training-cbs-multi"

# Output directory for configs
CONFIG_OUTPUT_DIR = "configs_branch_multi"

# =============================================================================
# GRADIENT ACCUMULATION SETTINGS
# =============================================================================

# Multi-GPU Configuration
NUM_GPUS = None  # Will auto-detect if None

# Maximum batch size PER GPU that fits in memory
MAX_MICRO_BATCH_SIZE_PER_GPU = 32  # Adjust based on your GPU

# For automatic estimation (if you set MAX_MICRO_BATCH_SIZE_PER_GPU = None)
MODEL_PARAMS = 45_000_000
GPU_MEMORY_GB = None  # Set to None to auto-detect

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def parse_checkpoint_tokens(checkpoint_step: str) -> int:
    """
    Parse checkpoint step identifier to get token count.
    
    Examples:
        "500M" -> 500,000,000
        "1000M" or "1B" -> 1,000,000,000
        "2500M" or "2.5B" -> 2,500,000,000
    
    Args:
        checkpoint_step: String like "500M", "1B", "2.5B"
    
    Returns:
        Token count as integer
    """
    checkpoint_step = checkpoint_step.upper().strip()
    
    # Try to match patterns like "500M", "1B", "2.5B"
    match = re.match(r'(\d+(?:\.\d+)?)\s*([MBK])', checkpoint_step)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step: {checkpoint_step}")
    
    value = float(match.group(1))
    unit = match.group(2)
    
    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
    }
    
    return int(value * multipliers[unit])


def generate_config_file(
    checkpoint_dir: str,
    checkpoint_file: str,
    checkpoint_step: str,
    k: int,
    seed: int,
    output_dir: str,
    grad_accum_config: aga.GradientAccumulationConfig
):
    """
    Generate a config file for a specific checkpoint-k-seed combination.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        checkpoint_file: Checkpoint filename
        checkpoint_step: Identifier for checkpoint (e.g., "100k")
        k: The scaling factor (batch_size = k * BASE_BATCH_SIZE)
        seed: Random seed for this branch
        output_dir: Directory to save the config file
        grad_accum_config: GradientAccumulationConfig with batch size settings
    """
    # Calculate scaled hyperparameters
    scaled_batch_size = k * BASE_BATCH_SIZE
    f_k = math.sqrt(k)
    scaled_lr = f_k * BASE_LEARNING_RATE
    scaled_lr_muon_adam = f_k * LEARNING_RATE_MUON_ADAM
    
    # Get batch configuration from grad_accum_config
    micro_batch_per_gpu = grad_accum_config.micro_batch_size_per_gpu
    grad_accum_steps = grad_accum_config.gradient_accumulation_steps
    num_gpus = grad_accum_config.num_gpus
    effective_batch = grad_accum_config.effective_batch_size
    
    # Max tokens: checkpoint starting position + window size
    checkpoint_tokens = parse_checkpoint_tokens(checkpoint_step)
    max_tokens = checkpoint_tokens + BRANCH_WINDOW_SIZE_TOKENS
    
    # Clean checkpoint name for file naming (remove .pt extension)
    ckpt_name = checkpoint_file.replace('.pt', '')
    
    # Output directory naming: include checkpoint step and seed
    branch_out_dir = f"out_branch_{ckpt_name}_k{k}_seed{seed}_{OPTIMIZER_TYPE}"
    
    # WandB run name: include checkpoint identifier and seed
    wandb_run_name = f"branch_{checkpoint_step}_k{k}_s{seed}_B{effective_batch}_lr{scaled_lr:.2e}_{OPTIMIZER_TYPE}"
    
    # Multi-GPU comments
    if num_gpus > 1:
        gpu_comment = f"""#   - Number of GPUs: {num_gpus}
#   - Micro-batch per GPU: {micro_batch_per_gpu}
#   - Total micro-batch per step: {micro_batch_per_gpu * num_gpus}
#   - Gradient accumulation: {grad_accum_steps} steps
#   - Effective batch size: {effective_batch} ({micro_batch_per_gpu}×{num_gpus}×{grad_accum_steps})"""
        effective_note = f"# NOTE: Effective batch = {micro_batch_per_gpu} (per GPU) × {num_gpus} (GPUs) × {grad_accum_steps} (accum) = {effective_batch}"
    else:
        gpu_comment = f"""#   - Micro-batch size: {micro_batch_per_gpu} (fits in GPU memory)
#   - Gradient accumulation: {grad_accum_steps} steps
#   - Effective batch size: {effective_batch}"""
        effective_note = f"# NOTE: Effective batch size = {micro_batch_per_gpu} × {grad_accum_steps} = {effective_batch}"
    
    # Generate config content
    config_content = f'''# Configuration for Branch: Checkpoint={checkpoint_step}, k={k}, seed={seed}
# Generated by generate_multi_checkpoint_configs.py
# 
# This branch trains with:
#   - Source checkpoint: {checkpoint_dir}/{checkpoint_file} (step: {checkpoint_step})
#   - Random seed: {seed}
#   - Target batch size: {scaled_batch_size} (k={k} × BASE_BATCH_SIZE={BASE_BATCH_SIZE})
{gpu_comment}
#   - Learning rate: {scaled_lr:.6f} (sqrt({k}) × BASE_LR={BASE_LEARNING_RATE})
#   - Checkpoint starting position: {checkpoint_tokens:,} tokens
#   - Training window size: {BRANCH_WINDOW_SIZE_TOKENS:,} tokens
#   - Training until: {max_tokens:,} tokens (start + window)

# I/O
out_dir = '{branch_out_dir}'
init_from = 'resume'
resume_ckpt_fname = '{checkpoint_dir}/{checkpoint_file}'
log_interval_tokens = {LOG_INTERVAL_TOKENS}
checkpoint_interval_tokens = {CHECKPOINT_INTERVAL_TOKENS}
eval_iters = {EVAL_ITERS}

# WandB logging
wandb_log = {WANDB_LOG}
wandb_project = '{WANDB_PROJECT}'
wandb_run_name = '{wandb_run_name}'

# Data
train_data_file = '{TRAIN_DATA_FILE}'
val_data_file = '{VAL_DATA_FILE}'
block_size = {BASE_BLOCK_SIZE}
checkpoint_token_pos = 0  # Will be overwritten by checkpoint's saved value
branch_seed = {seed}  # Offset data position: seed × window_size
branch_window_size_tokens = {BRANCH_WINDOW_SIZE_TOKENS}

# Random seed for dropout and stochasticity  
train_seed = {seed}  # Same seed for reproducibility

# Training - with gradient accumulation for memory efficiency
batch_size = {micro_batch_per_gpu}  # Micro-batch size per GPU (fits in GPU memory)
gradient_accumulation_steps = {grad_accum_steps}  # Accumulate to effective batch = {effective_batch}
max_tokens = {max_tokens}

{effective_note}
# This matches the target scaled batch size for k={k}

# Optimizer
optimizer_type = "{OPTIMIZER_TYPE}"
learning_rate = {scaled_lr}  # sqrt({k}) × BASE_LR={BASE_LEARNING_RATE}
weight_decay = {WEIGHT_DECAY}
beta1 = {BETA1}
beta2 = {BETA2}
grad_clip = 1.0
lr_muon_adam = {scaled_lr_muon_adam} # sqrt({k}) × BASE_LR={LEARNING_RATE_MUON_ADAM}
wd_muon_adam = {WEIGHT_DECAY_MUON_ADAM}

# Branch metadata (for tracking)
# checkpoint_identifier = "{checkpoint_step}"
# k_value = {k}
# seed = {seed}
# base_batch_size = {BASE_BATCH_SIZE}
# base_learning_rate = {BASE_LEARNING_RATE}
# scaling_factor_fk = {f_k}
# num_gpus = {num_gpus}
# max_micro_batch_size_per_gpu = {micro_batch_per_gpu}
# window_size_tokens = {BRANCH_WINDOW_SIZE_TOKENS}
'''
    
    # Save config file
    # Filename format: config_<checkpoint>_k<k>_seed<seed>.py
    config_filename = f"config_{ckpt_name}_k{k}_seed{seed}.py"
    config_path = os.path.join(output_dir, config_filename)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_filename


def main():
    """Generate all config files for the multi-checkpoint, multi-seed branched training experiment."""
    
    print("="*80)
    print("Multi-Checkpoint Multi-Seed Branched Training Config Generator")
    print("="*80)
    print()
    print("Base Configuration:")
    print(f"  Base batch size (B): {BASE_BATCH_SIZE}")
    print(f"  Base learning rate (eta): {BASE_LEARNING_RATE}")
    print(f"  Block size: {BASE_BLOCK_SIZE}")
    print(f"  Training tokens per branch: {DELTA_STEPS_AS_TOKENS:,}")
    print(f"  Window size: {BRANCH_WINDOW_SIZE_TOKENS:,} tokens")
    print()
    
    print("Checkpoints to branch from:")
    for i, (ckpt_dir, ckpt_file, ckpt_step) in enumerate(CHECKPOINTS, 1):
        print(f"  {i}. {ckpt_dir}/{ckpt_file} (step: {ckpt_step})")
    print()
    
    print(f"Seeds per checkpoint: {BRANCH_SEEDS}")
    print(f"K values: {K_VALUES}")
    print()
    
    # Calculate total number of configs
    total_configs = len(CHECKPOINTS) * len(BRANCH_SEEDS) * len(K_VALUES)
    print(f"Total configs to generate: {total_configs}")
    print(f"  = {len(CHECKPOINTS)} checkpoints × {len(BRANCH_SEEDS)} seeds × {len(K_VALUES)} k-values")
    print()
    
    # Detect or use specified number of GPUs
    if NUM_GPUS is not None:
        num_gpus = NUM_GPUS
        print(f"Using specified number of GPUs: {num_gpus}")
    else:
        num_gpus = aga.get_num_gpus()
        print(f"Detected {num_gpus} GPU(s)")
    print()
    
    # Determine max micro-batch size per GPU
    if MAX_MICRO_BATCH_SIZE_PER_GPU is not None:
        max_micro_batch_per_gpu = MAX_MICRO_BATCH_SIZE_PER_GPU
        print(f"Using specified max micro-batch size per GPU: {max_micro_batch_per_gpu}")
    else:
        print("Automatically estimating max micro-batch size per GPU...")
        
        gpu_mem = GPU_MEMORY_GB
        if gpu_mem is None:
            gpu_mem = aga.get_gpu_memory()
            if gpu_mem is not None:
                print(f"  Detected GPU memory: {gpu_mem:.1f} GB")
            else:
                print("  Could not detect GPU memory, using conservative estimate")
        
        max_micro_batch_per_gpu = aga.estimate_max_batch_size_from_memory(
            model_params=MODEL_PARAMS,
            sequence_length=BASE_BLOCK_SIZE,
            gpu_memory_gb=gpu_mem if gpu_mem is not None else 24
        )
        print(f"  Estimated max micro-batch size per GPU: {max_micro_batch_per_gpu}")
    
    if num_gpus > 1:
        print(f"  Max batch per step (no accumulation): {max_micro_batch_per_gpu * num_gpus}")
    
    print()
    
    # Print batch size configuration table
    print("Batch Size Configuration:")
    aga.print_batch_size_table(K_VALUES, BASE_BATCH_SIZE, max_micro_batch_per_gpu, num_gpus)
    print()
    
    # Create output directory for configs
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    print(f"Saving configs to: {CONFIG_OUTPUT_DIR}/")
    print()
    
    # Generate config for each checkpoint-seed-k combination
    print("Generating configs:")
    config_count = 0
    
    for checkpoint_dir, checkpoint_file, checkpoint_step in CHECKPOINTS:
        print(f"\n  Checkpoint: {checkpoint_step} ({checkpoint_dir}/{checkpoint_file})")
        
        for seed in BRANCH_SEEDS:
            print(f"    Seed {seed}:")
            
            for k in K_VALUES:
                target_batch = k * BASE_BATCH_SIZE
                grad_accum_config = aga.calculate_gradient_accumulation(
                    target_batch_size=target_batch,
                    max_micro_batch_size_per_gpu=max_micro_batch_per_gpu,
                    num_gpus=num_gpus,
                    ensure_divisible=True
                )
                
                config_filename = generate_config_file(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_file=checkpoint_file,
                    checkpoint_step=checkpoint_step,
                    k=k,
                    seed=seed,
                    output_dir=CONFIG_OUTPUT_DIR,
                    grad_accum_config=grad_accum_config
                )
                
                config_count += 1
                
                # Print compact summary
                micro_batch = grad_accum_config.micro_batch_size_per_gpu
                grad_accum = grad_accum_config.gradient_accumulation_steps
                effective = grad_accum_config.effective_batch_size
                
                if num_gpus > 1:
                    print(f"      k={k:2d}: {config_filename:40s} "
                          f"[{micro_batch:2d}×{num_gpus}×{grad_accum:2d} = {effective:4d}]")
                else:
                    print(f"      k={k:2d}: {config_filename:40s} "
                          f"[{micro_batch:2d}×{grad_accum:2d} = {effective:4d}]")
    
    print()
    print("="*80)
    print("Config generation complete!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  • Total configs generated: {config_count}")
    print(f"  • Configs saved to: {CONFIG_OUTPUT_DIR}/")
    print(f"  • Checkpoints: {len(CHECKPOINTS)}")
    print(f"  • Seeds per checkpoint: {len(BRANCH_SEEDS)}")
    print(f"  • K-values per seed: {len(K_VALUES)}")
    print(f"  • Number of GPUs: {num_gpus}")
    print(f"  • Max micro-batch size per GPU: {max_micro_batch_per_gpu}")
    if num_gpus > 1:
        print(f"  • Max batch per step (no accum): {max_micro_batch_per_gpu * num_gpus}")
    print()
    print("Next steps:")
    print(f"  1. Review the configs in '{CONFIG_OUTPUT_DIR}/'")
    print(f"  2. Ensure all checkpoints exist:")
    for ckpt_dir, ckpt_file, _ in CHECKPOINTS:
        print(f"     - {ckpt_dir}/{ckpt_file}")
    print(f"  3. Run: python run_multi_branch_sweep.py")
    print()
    print("Config file naming:")
    print("  Format: config_<checkpoint_name>_k<k>_seed<seed>.py")
    print("  Example: config_ckpt_k4_seed1.py")
    print()
    print("Output directory naming:")
    print("  Format: out_branch_<checkpoint_name>_k<k>_seed<seed>_<optimizer>")
    print("  Example: out_branch_ckpt_k4_seed1_adamw")
    print()


if __name__ == "__main__":
    main()

