"""
Shuffle binary token dataset at block level.
Streaming version - works with datasets larger than available RAM.
SAVES SHUFFLE INDICES for verification.

Usage:
    python shuffle_dataset.py --input train.bin --output train_shuffled.bin --block-size 1024 --seed 42
"""

import numpy as np
import argparse
import os


def shuffle_dataset_streaming(input_file: str, output_file: str, block_size: int = 1024, seed: int = 42):
    """
    Shuffle dataset at block level using streaming (doesn't require loading entire file).
    
    Strategy: Fisher-Yates shuffle that reads/writes blocks on demand.
    Saves shuffle indices to .npy file for validation.
    """
    
    print(f"\n{'='*70}")
    print(f"Block-Level Dataset Shuffling (Streaming with Index Saving)")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Block size: {block_size}")
    print(f"Random seed: {seed}")
    print()
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Data file not found: {input_file}")
    
    input_size = os.path.getsize(input_file)
    print(f"Input file size: {input_size:,} bytes ({input_size/1e9:.3f} GB)")
    
    dtype = np.uint16
    dtype_size = dtype().itemsize
    num_tokens = input_size // dtype_size
    num_blocks = num_tokens // block_size
    tokens_used = num_blocks * block_size
    tokens_trimmed = num_tokens - tokens_used
    
    print(f"Total tokens: {num_tokens:,}")
    print(f"Number of blocks: {num_blocks:,}")
    print(f"Tokens used: {tokens_used:,}")
    if tokens_trimmed > 0:
        print(f"Tokens trimmed: {tokens_trimmed:,} (last partial block removed)")
    print()
    
    # Open input file for reading
    input_data = np.memmap(input_file, dtype=dtype, mode='r')
    
    # Create output file
    output_data = np.memmap(output_file, dtype=dtype, mode='w+', shape=(tokens_used,))
    
    # Generate shuffle indices
    print("Generating shuffle indices...")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(num_blocks)
    
    # Save shuffle indices for later verification
    indices_file = output_file.replace('.bin', '_indices.npy')
    print(f"Saving indices to {indices_file}...")
    np.save(indices_file, shuffle_indices)
    print(f"✓ Saved {num_blocks:,} shuffle indices")
    
    # Write blocks in shuffled order
    print("\nWriting blocks in shuffled order...")
    for new_pos, old_block_idx in enumerate(shuffle_indices):
        if (new_pos + 1) % max(1, num_blocks // 10) == 0:
            progress_pct = 100 * (new_pos + 1) / num_blocks
            print(f"  Progress: {progress_pct:.1f}% ({new_pos + 1:,}/{num_blocks:,} blocks)")
        
        # Read old block
        old_start = old_block_idx * block_size
        old_end = old_start + block_size
        block = input_data[old_start:old_end]
        
        # Write to new position
        new_start = new_pos * block_size
        new_end = new_start + block_size
        output_data[new_start:new_end] = block
    
    output_data.flush()
    
    output_size = os.path.getsize(output_file)
    print(f"\nOutput file size: {output_size:,} bytes ({output_size/1e9:.3f} GB)")
    
    # ===== QUICK VERIFICATION USING INDICES =====
    print("\n" + "="*70)
    print("Verifying Shuffle Using Saved Indices")
    print("="*70)
    
    # Verify a sample using the indices
    sample_size = min(1000, num_blocks)
    mismatches = 0
    
    print(f"Verifying {sample_size:,} blocks...")
    for i in range(sample_size):
        old_block_idx = shuffle_indices[i]
        
        # Read from output
        output_start = i * block_size
        output_block = output_data[output_start:output_start + block_size]
        
        # Read from input at original position
        input_start = old_block_idx * block_size
        input_block = input_data[input_start:input_start + block_size]
        
        if not np.array_equal(output_block, input_block):
            mismatches += 1
    
    if mismatches == 0:
        print(f"✓ All {sample_size:,} sampled blocks verified")
        print(f"✓✓ Shuffling successful!")
    else:
        print(f"✗ {mismatches} mismatches found!")
        raise ValueError(f"Verification failed: {mismatches} mismatches")
    
    # Statistics using indices
    print("\n" + "="*70)
    print("Shuffle Statistics (from Indices)")
    print("="*70)
    
    sample_indices = shuffle_indices[:sample_size]
    distances = np.abs(sample_indices - np.arange(sample_size))
    blocks_in_same_pos = np.sum(sample_indices == np.arange(sample_size))
    
    print(f"Blocks in same position: {blocks_in_same_pos}/{sample_size} ({100*blocks_in_same_pos/sample_size:.2f}%)")
    print(f"Mean block distance: {np.mean(distances):,.0f}")
    print(f"Median block distance: {np.median(distances):,.0f}")
    print(f"Max block distance: {np.max(distances):,}")
    
    ideal_mean = num_blocks / 2
    print(f"(Ideal mean for perfect shuffle: ~{ideal_mean:,.0f})")
    
    if blocks_in_same_pos < max(1, num_blocks // 1000):  # < 0.1%
        print(f"✓ Excellent shuffle quality")
    
    print(f"\n{'='*70}")
    print(f"✓✓ Shuffling complete and verified!")
    print(f"{'='*70}\n")
    print(f"Files created:")
    print(f"  {output_file} ({output_size/1e9:.3f} GB)")
    print(f"  {indices_file} ({os.path.getsize(indices_file)/1e6:.2f} MB)")
    print(f"\nTo verify later, call validate with:")
    print(f"  validate_shuffling('{input_file}', '{output_file}', indices_file='{indices_file}')")
    
    # Cleanup
    del input_data
    del output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle binary token dataset at block level (streaming)')
    parser.add_argument('--input', type=str, required=True, help='Input binary file')
    parser.add_argument('--output', type=str, required=True, help='Output binary file')
    parser.add_argument('--block-size', type=int, default=1024, help='Tokens per block (default: 1024)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    shuffle_dataset_streaming(
        input_file=args.input,
        output_file=args.output,
        block_size=args.block_size,
        seed=args.seed
    )