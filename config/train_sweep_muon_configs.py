# I/O
out_dir = 'out_test_code_for_sweep_muon'
init_from = 'scratch'              # 'scratch' or 'resume'
# resume_ckpt_fname = "tokens(9.22e3)_tloss10.087_vloss9.981_ckpt.pt"
log_interval_tokens = 3072      # Log every 100K tokens
checkpoint_interval_tokens = 3072  # Checkpoint every 5M tokens
eval_iters = 1                   # Number of evaluation iterations (steps) not on a token basis

# WandB logging
wandb_log = True
wandb_project = 'hyperparam-search'
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
batch_size = 3  # Per-GPU batch size
gradient_accumulation_steps = 1
max_tokens = 9000  # Train for 1B tokens

# AdamW optimizer (no scheduler)
optimizer_type = "muon" # str: adamw or muon
learning_rate = 3e-4
weight_decay = 0.1
# beta1 = 0.9
# beta2 = 0.95