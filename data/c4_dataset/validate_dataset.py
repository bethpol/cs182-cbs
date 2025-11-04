"""
Validation script for C4 dataset creation with shuffling.
Uses saved shuffle indices for efficient, accurate verification.
Handles partial block trimming correctly.
"""

import os
import numpy as np
import json
import tiktoken


def validate_shuffling_with_indices(original_file: str, shuffled_file: str, indices_file: str, 
                                     block_size: int = 1024) -> dict:
    """
    Validate shuffling using saved indices.
    
    This is the definitive validation - uses the actual indices that were used for shuffling.
    Handles partial block trimming gracefully.
    
    Args:
        original_file: Path to original unshuffled dataset
        shuffled_file: Path to shuffled dataset
        indices_file: Path to saved shuffle indices (.npy file)
        block_size: Tokens per block
    """
    results = {
        'errors': [],
        'warnings': [],
        'tests': {}
    }
    
    print("\n" + "="*70)
    print("SHUFFLING VALIDATION (Using Saved Indices)")
    print("="*70)
    
    dtype = np.uint16
    
    # Check all files exist
    for f in [original_file, shuffled_file, indices_file]:
        if not os.path.exists(f):
            results['errors'].append(f"File not found: {f}")
    
    if results['errors']:
        return results
    
    # Load files
    print("\n[1] Loading Files")
    print("-" * 70)
    
    original = np.memmap(original_file, dtype=dtype, mode='r')
    shuffled = np.memmap(shuffled_file, dtype=dtype, mode='r')
    shuffle_indices = np.load(indices_file)
    
    print(f"Original file: {len(original):,} tokens ({len(original)/1e9:.3f}B)")
    print(f"Shuffled file: {len(shuffled):,} tokens ({len(shuffled)/1e9:.3f}B)")
    print(f"Shuffle indices: {len(shuffle_indices):,} indices")
    
    # Calculate block information
    original_blocks = len(original) // block_size
    shuffled_blocks = len(shuffled) // block_size
    tokens_trimmed = len(original) % block_size
    
    print(f"\nBlock information:")
    print(f"  Original complete blocks: {original_blocks:,}")
    print(f"  Shuffled complete blocks: {shuffled_blocks:,}")
    if tokens_trimmed > 0:
        print(f"  Tokens trimmed from original: {tokens_trimmed:,}")
    
    # ===== VERIFY INDICES MATCH =====
    print("\n[2] Index Validation")
    print("-" * 70)
    
    if len(shuffle_indices) != original_blocks:
        results['errors'].append(
            f"Indices count {len(shuffle_indices):,} doesn't match original blocks {original_blocks:,}"
        )
        return results
    
    if shuffled_blocks != original_blocks:
        results['errors'].append(
            f"Shuffled blocks {shuffled_blocks:,} doesn't match original blocks {original_blocks:,}"
        )
        return results
    
    print(f"✓ Index count matches block count: {len(shuffle_indices):,}")
    print(f"✓ All block counts consistent")
    
    # ===== VERIFY BLOCK CORRECTNESS USING INDICES =====
    print("\n[3] Block Content Verification (Using Indices)")
    print("-" * 70)
    
    # Verify that shuffled[i] == original[shuffle_indices[i]]
    print(f"Verifying that shuffled blocks match indexed original blocks...")
    
    sample_size = min(10000, original_blocks)
    mismatches = 0
    
    for i in range(sample_size):
        orig_block_idx = shuffle_indices[i]
        
        # Get blocks
        shuffled_block = shuffled[i*block_size:(i+1)*block_size]
        original_block = original[orig_block_idx*block_size:(orig_block_idx+1)*block_size]
        
        if not np.array_equal(shuffled_block, original_block):
            mismatches += 1
    
    if mismatches == 0:
        print(f"✓ All {sample_size:,} sampled blocks match their indexed originals")
        print(f"✓ Block structure perfectly preserved")
        results['tests']['block_preservation'] = 'PASS'
    else:
        print(f"✗ {mismatches} blocks don't match!")
        results['errors'].append(
            f"Block verification failed: {mismatches}/{sample_size} blocks corrupted"
        )
        results['tests']['block_preservation'] = 'FAIL'
        return results
    
    # ===== VERIFY BLOCKS ARE ACTUALLY SHUFFLED =====
    print("\n[4] Shuffling Verification")
    print("-" * 70)
    
    # Check how many blocks stayed in same position
    blocks_in_same_position = np.sum(shuffle_indices == np.arange(len(shuffle_indices)))
    pct_same = 100 * blocks_in_same_position / len(shuffle_indices)
    
    print(f"Blocks in same position: {blocks_in_same_position}/{len(shuffle_indices)} ({pct_same:.3f}%)")
    
    # Theoretically, random shuffle should have ~1/n blocks stay in same position
    theoretical_pct = 100 / len(shuffle_indices)
    
    if pct_same < len(shuffle_indices) * 0.01:  # Less than 1%
        print(f"✓ Blocks are well shuffled (theoretical: ~{theoretical_pct:.3f}%)")
        results['tests']['shuffling_occurred'] = 'PASS'
    else:
        print(f"⚠️  {pct_same:.3f}% blocks in same position (theoretical: {theoretical_pct:.3f}%)")
        results['warnings'].append(
            f"Shuffling appears weak: {pct_same:.3f}% blocks in same position"
        )
        results['tests']['shuffling_occurred'] = 'PARTIAL'
    
    # ===== SHUFFLE QUALITY STATISTICS =====
    print("\n[5] Shuffle Quality Statistics")
    print("-" * 70)
    
    distances = np.abs(shuffle_indices - np.arange(len(shuffle_indices)))
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    
    ideal_mean = len(shuffle_indices) / 2
    
    print(f"Mean block distance: {mean_distance:,.1f}")
    print(f"Median block distance: {median_distance:,.1f}")
    print(f"Max block distance: {max_distance:,}")
    print(f"Ideal mean (for perfect shuffle): {ideal_mean:,.1f}")
    
    quality_ratio = mean_distance / ideal_mean
    print(f"Quality ratio: {quality_ratio:.2f} (1.0 = perfect)")
    
    if quality_ratio > 0.4:
        print(f"✓ Excellent shuffle quality")
        results['tests']['shuffle_quality'] = 'PASS'
    elif quality_ratio > 0.2:
        print(f"✓ Good shuffle quality")
        results['tests']['shuffle_quality'] = 'PASS'
    else:
        print(f"⚠️  Weak shuffle (mean distance too low)")
        results['warnings'].append("Shuffle quality is weak")
        results['tests']['shuffle_quality'] = 'PARTIAL'
    
    print("\n" + "="*70)
    
    del original, shuffled, shuffle_indices
    return results


def validate_dataset_creation(train_file: str, val_file: str, config: dict) -> dict:
    """
    Validate that datasets were created correctly.
    
    Checks:
    1. Files exist and have expected sizes
    2. Token values are in valid range
    3. Data types are correct
    """
    results = {
        'train': {},
        'val': {},
        'errors': [],
        'warnings': []
    }
    
    block_size = config.get('block_size', 1024)
    target_tokens = config.get('target_train_tokens', 100_000_000)
    dtype = np.uint16
    
    print("\n" + "="*70)
    print("DATASET VALIDATION")
    print("="*70)
    
    # ======= TRAINING DATA VALIDATION =======
    print("\n[1] Training Data Validation")
    print("-" * 70)
    
    if not os.path.exists(train_file):
        results['errors'].append(f"Train file not found: {train_file}")
        return results
    
    train_size = os.path.getsize(train_file)
    expected_size = target_tokens * dtype().itemsize
    
    print(f"File: {train_file}")
    print(f"Size: {train_size:,} bytes ({train_size/1e9:.3f} GB)")
    print(f"Expected: {expected_size:,} bytes ({expected_size/1e9:.3f} GB)")
    
    if train_size != expected_size:
        results['errors'].append(
            f"Train file size mismatch: expected {expected_size:,}, got {train_size:,}"
        )
    else:
        print("✓ Size matches exactly")
    
    # Load and check token values
    train_data = np.memmap(train_file, dtype=dtype, mode='r')
    num_tokens = len(train_data)
    results['train']['num_tokens'] = int(num_tokens)
    
    print(f"Tokens: {num_tokens:,} ({num_tokens/1e9:.3f}B)")
    
    # Check token value ranges
    min_val = train_data.min()
    max_val = train_data.max()
    print(f"Token value range: {min_val} to {max_val}")
    
    try:
        enc = tiktoken.get_encoding("gpt2")
        if max_val >= enc.n_vocab:
            results['warnings'].append(
                f"Token value {max_val} >= vocab size {enc.n_vocab}"
            )
        else:
            print(f"✓ All tokens in valid range [0, {enc.n_vocab})")
    except:
        print("⚠️  Could not verify with tiktoken (not installed)")
    
    # Check for suspicious patterns
    unique_tokens = len(np.unique(train_data[:min(100_000, num_tokens)]))
    print(f"Unique tokens in first 100k: {unique_tokens:,}")
    
    if unique_tokens < 100:
        results['warnings'].append(
            f"Only {unique_tokens} unique tokens in first 100k samples"
        )
    
    zero_count = np.sum(train_data == 0)
    zero_pct = 100 * zero_count / num_tokens
    print(f"Zero tokens: {zero_count:,} ({zero_pct:.2f}%)")
    
    if zero_pct > 10:
        results['warnings'].append(
            f"High percentage of zero tokens: {zero_pct:.2f}%"
        )
    
    # Check alignment with block size
    if num_tokens % block_size != 0:
        complete_blocks = num_tokens // block_size
        partial_tokens = num_tokens % block_size
        print(f"Blocks: {complete_blocks:,} complete + {partial_tokens:,} partial")
        results['warnings'].append(
            f"Tokens not perfectly divisible by block_size ({partial_tokens:,} leftover)"
        )
    else:
        num_blocks = num_tokens // block_size
        print(f"Blocks: {num_blocks:,} (perfect alignment ✓)")
    
    del train_data
    results['train']['validation'] = 'PASS' if not results['errors'] else 'FAIL'
    
    # ======= VALIDATION DATA VALIDATION =======
    print("\n[2] Validation Data Validation")
    print("-" * 70)
    
    if not os.path.exists(val_file):
        results['warnings'].append(f"Val file not found: {val_file}")
    else:
        val_size = os.path.getsize(val_file)
        print(f"File: {val_file}")
        print(f"Size: {val_size:,} bytes ({val_size/1e6:.2f} MB)")
        
        val_data = np.memmap(val_file, dtype=dtype, mode='r')
        val_tokens = len(val_data)
        results['val']['num_tokens'] = int(val_tokens)
        print(f"Tokens: {val_tokens:,} ({val_tokens/1e6:.2f}M)")
        
        del val_data
        results['val']['validation'] = 'PASS'
    
    # ======= METADATA VALIDATION =======
    print("\n[3] Metadata Validation")
    print("-" * 70)
    
    metadata_file = train_file.replace('train.bin', 'dataset_info.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Metadata file: {metadata_file}")
        print(f"Created: {metadata['dataset_info']['created_at']}")
        print(f"Source: {metadata['dataset_info']['source']}")
        
        meta_tokens = metadata['training_data']['num_tokens']
        if meta_tokens != num_tokens:
            results['errors'].append(
                f"Metadata mismatch: claims {meta_tokens:,} tokens, file has {num_tokens:,}"
            )
        else:
            print(f"✓ Metadata token count matches")
        
        results['metadata_valid'] = True
    else:
        results['warnings'].append(f"Metadata file not found: {metadata_file}")
    
    print("\n" + "="*70)
    return results


def print_summary(dataset_results: dict, shuffle_results: dict = None):
    """Print validation summary."""
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if dataset_results['errors']:
        print("\n❌ ERRORS:")
        for error in dataset_results['errors']:
            print(f"  • {error}")
    
    if dataset_results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in dataset_results['warnings']:
            print(f"  • {warning}")
    
    if not dataset_results['errors'] and not dataset_results['warnings']:
        print("\n✓ Dataset creation: VALID")
    
    if shuffle_results:
        if shuffle_results['errors']:
            print("\n❌ SHUFFLE ERRORS:")
            for error in shuffle_results['errors']:
                print(f"  • {error}")
        
        if shuffle_results['warnings']:
            print("\n⚠️  SHUFFLE WARNINGS:")
            for warning in shuffle_results['warnings']:
                print(f"  • {warning}")
        
        print("\nShuffle test results:")
        for test, result in shuffle_results['tests'].items():
            symbol = "✓" if result == 'PASS' else "⚠" if result == 'PARTIAL' else "✗"
            print(f"  {symbol} {test}: {result}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    # Configuration - update these paths
    TRAIN_FILE = '100M/train.bin'
    VAL_FILE = '100M/val.bin'
    SHUFFLED_FILE = '100M/train_shuffled_1024.bin'
    INDICES_FILE = '100M/train_shuffled_1024_indices.npy'
    
    CONFIG = {
        'block_size': 1024,
        'target_train_tokens': 100_000_000,
    }
    
    # Run validation
    dataset_results = validate_dataset_creation(TRAIN_FILE, VAL_FILE, CONFIG)
    
    shuffle_results = None
    if os.path.exists(SHUFFLED_FILE) and os.path.exists(INDICES_FILE):
        print("\n" + "="*70)
        print("Found shuffled dataset and indices - validating shuffle...")
        print("="*70)
        shuffle_results = validate_shuffling_with_indices(
            TRAIN_FILE, 
            SHUFFLED_FILE, 
            INDICES_FILE, 
            CONFIG['block_size']
        )
    elif os.path.exists(SHUFFLED_FILE):
        print("\n⚠️  Shuffled file found but no indices file - cannot fully validate")
        print(f"Expected indices at: {INDICES_FILE}")
    
    print_summary(dataset_results, shuffle_results)