"""
This script implements the "Branched Training to Measure CBS" experiment 
described in the user's screenshot.
"""

import math
import time
import torch
from torch import optim
import os
import sys
import numpy as np

# Path Setup
# Add the parent directory to the Python path
# This allows us to import model_45M, which we assume is in the parent dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

try:
    from model_45M import build_gpt_45m, count_parameters
except ImportError:
    print(f"Error: Could not import 'build_gpt_45m' from 'model_45M.py'.")
    print(f"Please ensure 'model_45M.py' is in the parent directory: {PARENT_DIR}")
    sys.exit(1)

# Adjusted Paths 
# Paths are relative to the parent directory
DATA_DIR = os.path.join(PARENT_DIR, "data/shakespeare_char")
train_bin = os.path.join(DATA_DIR, "train.bin")
val_bin   = os.path.join(DATA_DIR, "val.bin")

BASE_CHECKPOINT_PATH = os.path.join(PARENT_DIR, "out_trial_45m/ckpt.pt")

# Data Loader (Copied from train_trial_45M.py


def load_bin(path):
    """Load uint16 token ids from .bin file safely using NumPy."""
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        print("Please run the data preparation script first.")
        sys.exit(1)
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))

# Load data once to avoid repeated file I/O
try:
    train_data = load_bin(train_bin)
    val_data   = load_bin(val_bin)
    print(f"Loaded training data ({len(train_data):,} tokens)")
    print(f"Loaded validation data ({len(val_data):,} tokens)")
except SystemExit:
    sys.exit(1) # Exit if load_bin failed

def get_batch(split, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

# Main Experiment Function

def run_experiment():
    """
    Runs the branched training experiment.
    """
    
    # Experiment Hyperparameters
    # Base parameters from train_trial_45M.py
    BASE_BLOCK_SIZE = 128     # Must match the trial script
    BASE_BATCH_SIZE_B = 8       # This is B
    BASE_LR_ETA = 3e-4    # This is eta

    # Experiment parameters
    # These are the 'k' values to test
    K_VALUES = [1, 2, 4, 8, 16, 32] 
    # 'Delta' - small number of steps to train each branch
    DELTA_STEPS = 20
    # 'Smoothed loss' - we average the loss over the last N steps
    SMOOTHING_START_STEP = DELTA_STEPS // 2 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running experiment on device: {DEVICE}")

    # Check that the base checkpoint exists
    if not os.path.exists(BASE_CHECKPOINT_PATH):
        print(f"Error: Base checkpoint not found at {BASE_CHECKPOINT_PATH}")
        print("Please run 'python train_trial_45M.py' first to generate the checkpoint.")
        return

    print(f"Loading base checkpoint from: {BASE_CHECKPOINT_PATH}")
    # Load checkpoint data (this is just a dictionary of tensors)
    base_ckpt = torch.load(BASE_CHECKPOINT_PATH, map_location=DEVICE)

    results = {} # To store k: L_k

    # Loop over all branches k
    for k in K_VALUES:
        start_time = time.time()
        print(f"\n Running branch k={k}")

        # 1. Calculate new hyperparameters
        current_batch_size = k * BASE_BATCH_SIZE_B
        f_k = math.sqrt(k) # f(k) = sqrt(k) for Adam/AdamW
        current_lr = f_k * BASE_LR_ETA

        print(f"  Batch Size (k*B): {current_batch_size} (k={k})")
        print(f"  Learning Rate (f(k)*eta): {current_lr:.2e} (f(k)={f_k:.2f})")

        # 2. Build model
        model = build_gpt_45m(DEVICE)
        
        # 3. IMPORTANT: Resize position embeddings *before* loading state_dict.
        # The base model is initialized with block_size 1024, but the checkpoint
        # was saved with the smaller block_size from the trial script (128).
        if BASE_BLOCK_SIZE != model.config.block_size:
            print(f"  Resizing model's position embeddings from {model.config.block_size} to {BASE_BLOCK_SIZE} to match checkpoint.")
            model.transformer.wpe.weight = torch.nn.Parameter(
                model.transformer.wpe.weight[:BASE_BLOCK_SIZE].clone()
            )
            model.config.block_size = BASE_BLOCK_SIZE
        
        # 4. Load model from base checkpoint
        try:
            model.load_state_dict(base_ckpt)
            print(f"  Successfully loaded checkpoint into model.")
        except RuntimeError as e:
            print("\n--- ERROR ---")
            print("Failed to load state_dict. This can happen if the model architecture in 'model_45M.py'")
            print("does not match the architecture used to save the 'ckpt.pt' file.")
            print(f"Original error: {e}")
            print("Skipping this k value...\n")
            continue # Skip this k value
        
        model.train() # Set to training mode

        # 5. Create new optimizer with scaled LR
        opt = optim.AdamW(model.parameters(), lr=current_lr, betas=(0.9, 0.95), weight_decay=0.1)

        # 6. Run training for 'Delta' steps
        step_losses = []
        for step in range(DELTA_STEPS):
            # Get a batch
            x, y = get_batch("train", BASE_BLOCK_SIZE, current_batch_size, DEVICE)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass and optimization
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clipping
            opt.step()

            # Record loss for smoothing
            if step >= SMOOTHING_START_STEP:
                step_losses.append(loss.item())
            
            if (step+1) % 5 == 0 or step == DELTA_STEPS - 1:
                print(f"  k={k}, step {step+1:3d}/{DELTA_STEPS}, loss {loss.item():.4f}")

        # 7. Calculate smoothed loss L_k
        if len(step_losses) > 0:
            L_k = sum(step_losses) / len(step_losses)
        else:
            L_k = float('nan') # Should not happen if DELTA_STEPS > 0
        
        end_time = time.time()
        print(f"--- k={k} finished in {end_time - start_time:.2f}s. Smoothed Loss (L_k) = {L_k:.4f} ---")
        
        results[k] = L_k

    # Print final report
    print("\n" + "="*50)
    print("  Branched Training Experiment Complete")
    print("="*50)
    print(f"Base B = {BASE_BATCH_SIZE_B}, Base eta = {BASE_LR_ETA}")
    print(f"Trained for {DELTA_STEPS} steps, smoothed over last {DELTA_STEPS - SMOOTHING_START_STEP} steps.")
    print("\nResults (k: L_k):")
    print("---")
    print(f"{'k':<4} | {'Batch Size':<10} | {'LR (f(k)*eta)':<15} | {'Smoothed Loss (L_k)':<20}")
    print(f"{'-'*4} | {'-'*10} | {'-'*15} | {'-'*20}")
    
    for k, loss in results.items():
        batch = k * BASE_BATCH_SIZE_B
        lr = math.sqrt(k) * BASE_LR_ETA
        print(f"{k:<4} | {batch:<10} | {lr:<15.4e} | {loss:<20.4f}")
    print("---")

if __name__ == "__main__":
    run_experiment()
