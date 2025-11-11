# I/O
out_dir = 'out_c4Dataset_hyperparamSearch_muon_lr(1e-3)_wd(0.003)_adamw_lr(3e-4)_wd(0.003)'
init_from = 'scratch'              # 'scratch' or 'resume'
# resume_ckpt_fname = "checkpoint_last_run1.pt"
log_interval_tokens = 500_000      # 500_000 Log every 100K tokens
checkpoint_interval_tokens = 5_000_000  # 5_000_000 Checkpoint every 5M tokens
eval_iters = 152                  # Number of evaluation iterations (steps) not on a token basis

# WandB logging
wandb_log = True
wandb_project = 'hyperparam-search'
wandb_run_name = f"{out_dir}"

# Data
train_data_file = 'data/c4_dataset/100M/train_shuffled_512.bin'
val_data_file = 'data/c4_dataset/100M/val.bin'
block_size = 512
# total_tokens = None  # Auto-detect from file
checkpoint_token_pos = 0  # Starting position (for resume)
branch_seed = -1  # No branching by default
branch_window_size_tokens = 100000000  # Required parameter

# Training
batch_size = 32  # Per-GPU batch size
gradient_accumulation_steps = 1
max_tokens = 100_000_000  # 100_000_000 Train for 1B tokens

# AdamW optimizer (no scheduler)
optimizer_type = "muon" # str: adamw or muon
learning_rate = 1e-3
weight_decay = 0.003
lr_muon_adam = 3e-4
wd_muon_adam = 0.003
# beta1 = 0.9
# beta2 = 0.95