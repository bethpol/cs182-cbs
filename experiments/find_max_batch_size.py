"""
Find Maximum Batch Size for Your GPU

This script empirically tests increasing batch sizes to find the maximum
that fits in your GPU memory. Use this to set MAX_MICRO_BATCH_SIZE accurately.

Usage:
    python find_max_batch_size.py [--model-config path/to/model_config.py]
    
If no model config is provided, tests with a dummy model based on parameters below.
"""

import argparse
import sys
import torch
import torch.nn as nn


# =============================================================================
# TEST CONFIGURATION - Modify to match your model
# =============================================================================

# Model architecture (for dummy model if actual model not provided)
MODEL_PARAMS = 45_000_000  # ~45M parameters
VOCAB_SIZE = 50257
EMBED_DIM = 512
NUM_LAYERS = 12
NUM_HEADS = 8
SEQUENCE_LENGTH = 128

# Test parameters
START_BATCH_SIZE = 1
MAX_BATCH_SIZE = 512
BATCH_SIZE_STEP = "double"  # "double" or integer step size

# Number of iterations to test (to ensure stable memory usage)
NUM_TEST_ITERS = 3

# =============================================================================


class DummyTransformer(nn.Module):
    """
    Simplified transformer for testing batch size limits.
    
    This mimics the memory usage of a real transformer without requiring
    the full model code.
    """
    
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Embeddings
        x = self.embed(idx)
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss (to trigger backprop)
        # Shift for next-token prediction
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets = idx[:, 1:].reshape(-1)
        loss = nn.functional.cross_entropy(logits_flat, targets)
        
        return loss


def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_batch_size(model, optimizer, batch_size, seq_len, vocab_size, num_iters=3):
    """
    Test if a given batch size works.
    
    Returns:
        (success, peak_memory_mb, error_msg)
    """
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        for i in range(num_iters):
            # Create random batch
            idx = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
            
            # Forward pass
            optimizer.zero_grad()
            loss = model(idx)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Record peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return True, peak_memory, None
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, None, "OOM"
        else:
            return False, None, str(e)
    
    except Exception as e:
        return False, None, str(e)


def find_max_batch_size(
    model,
    optimizer,
    seq_len,
    vocab_size,
    start_batch=1,
    max_batch=512,
    step="double",
    num_iters=3
):
    """
    Binary search to find maximum batch size.
    
    Returns:
        (max_batch_size, memory_usage_dict)
    """
    print(f"\n{'='*80}")
    print("Finding Maximum Batch Size")
    print(f"{'='*80}\n")
    
    results = {}
    
    # Phase 1: Exponential growth to find upper bound
    print("Phase 1: Finding upper bound...")
    batch_size = start_batch
    last_successful = start_batch
    
    while batch_size <= max_batch:
        print(f"\nTesting batch_size = {batch_size}...", end=" ", flush=True)
        
        success, peak_mem, error = test_batch_size(
            model, optimizer, batch_size, seq_len, vocab_size, num_iters
        )
        
        if success:
            print(f"✓ SUCCESS (peak memory: {peak_mem:.1f} MB)")
            results[batch_size] = peak_mem
            last_successful = batch_size
            
            # Increase batch size
            if step == "double":
                batch_size *= 2
            else:
                batch_size += step
        else:
            print(f"✗ FAILED ({error})")
            break
    
    # Phase 2: Binary search to find exact maximum
    print(f"\nPhase 2: Binary search between {last_successful} and {batch_size}...")
    
    lower = last_successful
    upper = batch_size
    
    while upper - lower > 1:
        mid = (lower + upper) // 2
        
        print(f"\nTesting batch_size = {mid}...", end=" ", flush=True)
        
        success, peak_mem, error = test_batch_size(
            model, optimizer, mid, seq_len, vocab_size, num_iters
        )
        
        if success:
            print(f"✓ SUCCESS (peak memory: {peak_mem:.1f} MB)")
            results[mid] = peak_mem
            lower = mid
        else:
            print(f"✗ FAILED ({error})")
            upper = mid
    
    max_batch_size = lower
    
    return max_batch_size, results


def print_results(max_batch_size, results, gpu_name, gpu_memory_gb):
    """Print summary of results."""
    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}\n")
    
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    print(f"\nMaximum batch size: {max_batch_size}")
    print(f"\nMemory usage by batch size:")
    print(f"{'Batch Size':<12} | {'Peak Memory (MB)':<20}")
    print(f"{'-'*12} | {'-'*20}")
    
    for batch_size in sorted(results.keys()):
        mem = results[batch_size]
        print(f"{batch_size:<12} | {mem:<20.1f}")
    
    print(f"\n{'='*80}")
    print("Recommendation")
    print(f"{'='*80}\n")
    print(f"Set this in generate_branch_configs_with_grad_accum.py:")
    print(f"\n    MAX_MICRO_BATCH_SIZE = {max_batch_size}\n")
    print(f"This will use {results[max_batch_size]:.1f} MB "
          f"({results[max_batch_size]/1024:.2f} GB) of GPU memory.")
    print(f"Total GPU memory: {gpu_memory_gb:.1f} GB")
    print(f"Utilization: {results[max_batch_size]/1024/gpu_memory_gb*100:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Find maximum batch size for your GPU")
    parser.add_argument("--model-config", type=str, help="Path to model config (optional)")
    parser.add_argument("--start-batch", type=int, default=START_BATCH_SIZE)
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"{'='*80}")
    print("GPU Batch Size Test")
    print(f"{'='*80}")
    print(f"\nGPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    print()
    
    # Create model
    if args.model_config:
        print(f"Loading model from config: {args.model_config}")
        # TODO: Load actual model from config
        print("ERROR: Custom model loading not yet implemented.")
        print("Using dummy model instead...")
        model = None
    
    if not args.model_config or model is None:
        print(f"Creating dummy transformer:")
        print(f"  Parameters: ~{MODEL_PARAMS/1e6:.1f}M")
        print(f"  Embed dim: {EMBED_DIM}")
        print(f"  Layers: {NUM_LAYERS}")
        print(f"  Heads: {NUM_HEADS}")
        print(f"  Sequence length: {args.seq_len}")
        
        model = DummyTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            max_seq_len=args.seq_len
        ).cuda()
        
        actual_params = count_parameters(model)
        print(f"  Actual parameters: {actual_params/1e6:.1f}M")
    
    # Create optimizer (AdamW with typical settings)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    print("\nStarting batch size tests...")
    print("This may take a few minutes...")
    
    # Find maximum batch size
    max_batch_size, results = find_max_batch_size(
        model=model,
        optimizer=optimizer,
        seq_len=args.seq_len,
        vocab_size=VOCAB_SIZE,
        start_batch=args.start_batch,
        max_batch=args.max_batch,
        step=BATCH_SIZE_STEP,
        num_iters=NUM_TEST_ITERS
    )
    
    # Print results
    print_results(max_batch_size, results, gpu_name, gpu_memory_gb)


if __name__ == "__main__":
    main()


