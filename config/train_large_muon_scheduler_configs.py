# I/O
out_dir = 'out_c4Dataset_largeRun_muon_schedulerTest_2phase'
init_from = 'scratch'              # 'scratch' or 'resume'
# resume_ckpt_fname = "checkpoint.pt"
log_interval_tokens = 100_000      # 500_000 Log every 100K tokens
checkpoint_interval_tokens = 500_000  # 5_000_000 Checkpoint every 5M tokens
eval_iters = 152                  # Number of evaluation iterations (steps) not on a token basis

# WandB logging
wandb_log = True
wandb_project = 'cbs-train'
wandb_run_name = f"{out_dir}"

# Data
train_data_file = 'data/c4_dataset/100M/train_shuffled_1024.bin'
val_data_file = 'data/c4_dataset/100M/val.bin'
block_size = 1024
# total_tokens = None  # Auto-detect from file
checkpoint_token_pos = 0  # Starting position (for resume)
branch_seed = -1  # No branching by default
branch_window_size_tokens = 100000000  # Required parameter

# Training
batch_size = 32  # Per-GPU batch size
gradient_accumulation_steps = 1
max_tokens = 2_000_000  # 100_000_000 Train for 1B tokens

# AdamW or Muon optimizer
optimizer_type = "muon" # str: adamw or muon
learning_rate = 1e-4
weight_decay = 0.01
lr_muon_adam = 3e-4
wd_muon_adam = 0.1
# beta1 = 0.9
# beta2 = 0.95

# Learning rate scheduler (unified 2-phase or 3-phase)
use_scheduler = True           # Set to True to enable
warmup_tokens = 500_000      # Warmup phase
stable_tokens = 0     # Stable phase for a 3phase setup. set to 0 for 2-phase i.e. if you don't want a stable phase and want to go to the decay phase directly after warmup
min_lr_factor = 0.1                  # Minimum LR factor in range [0,1]
