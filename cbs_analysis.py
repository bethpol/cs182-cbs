#!/usr/bin/env python3
"""
Extract training losses from branch training logs and save to CSV.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional
import csv


# Valid k-values to include in analysis and tables
VALID_K_VALUES = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]


def extract_info_from_log(log_file: str) -> Optional[Dict]:
    """
    Extract key information from a training log file.
    First tries to parse from filename, then from file content.
    
    Returns:
        Dictionary with checkpoint, k_value, seed, final_train_loss, final_val_loss
    """
    try:
        # Try to extract from filename first
        filename = os.path.basename(log_file)
        
        checkpoint_from_file = None
        k_value_from_file = None
        seed_from_file = None
        
        # Extract from filename
        filename_pattern = r'branch_(\w+)_k([\d\.]+)_seed(\d+)'
        filename_match = re.search(filename_pattern, filename)
        if filename_match:
            checkpoint_from_file = filename_match.group(1)
            k_value_from_file = float(filename_match.group(2))
            # Convert to int if whole number
            if k_value_from_file == int(k_value_from_file):
                k_value_from_file = int(k_value_from_file)
            seed_from_file = int(filename_match.group(3))
        
        # Try UTF-8, fall back to latin-1 for special characters
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(log_file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Extract checkpoint - prefer filename
        checkpoint = checkpoint_from_file
        if not checkpoint:
            checkpoint_match = re.search(r'Checkpoint:\s+(\w+)', content)
            if checkpoint_match:
                checkpoint = checkpoint_match.group(1)
        if not checkpoint:
            return None
        
        # Extract k-value - prefer filename
        k_value = k_value_from_file
        if k_value is None:
            k_match = re.search(r'K-value:\s+([\d\.]+)', content)
            if k_match:
                k_value = float(k_match.group(1))
                if k_value == int(k_value):
                    k_value = int(k_value)
        if k_value is None:
            return None
        
        # Extract seed - prefer filename
        seed = seed_from_file
        if seed is None:
            seed_match = re.search(r'Seed:\s+(\d+)', content)
            if seed_match:
                seed = int(seed_match.group(1))
        if seed is None:
            return None
        
        # Extract final training loss
        train_loss_pattern = r'tokens:\s+[\d,]+\s+\|\s+loss:\s+([\d.]+)\s+\|'
        train_losses = re.findall(train_loss_pattern, content)
        if not train_losses:
            return None
        final_train_loss = float(train_losses[-1])
        
        # Extract final validation loss
        val_loss_pattern = r'Final validation loss:\s+([\d.]+)'
        val_match = re.search(val_loss_pattern, content)
        final_val_loss = float(val_match.group(1)) if val_match else None
        
        # If no "Final validation loss", get the last validation loss reported
        if final_val_loss is None:
            val_losses = re.findall(r'Validation loss:\s+([\d.]+)', content)
            if val_losses:
                final_val_loss = float(val_losses[-1])
        
        # Filter: only include if k_value is in VALID_K_VALUES
        if k_value not in VALID_K_VALUES:
            return None
        
        return {
            'checkpoint': checkpoint,
            'k_value': k_value,
            'seed': seed,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'log_file': log_file
        }
    
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return None


def collect_results_from_logs(log_directory: str) -> Dict[str, List[Dict]]:
    """
    Collect results from all log files in a directory.
    Only includes k-values in VALID_K_VALUES list.
    
    Returns:
        Dictionary mapping checkpoint -> list of results
    """
    results_by_checkpoint = {}
    
    log_files = list(Path(log_directory).rglob("*.log")) + \
                list(Path(log_directory).rglob("*.txt"))
    
    if not log_files:
        print(f"  No .log or .txt files found in {log_directory}")
        return results_by_checkpoint
    
    print(f"  Found {len(log_files)} log files")
    successful = 0
    failed = 0
    filtered = 0
    
    for log_file in log_files:
        info = extract_info_from_log(str(log_file))
        if info:
            checkpoint = info['checkpoint']
            if checkpoint not in results_by_checkpoint:
                results_by_checkpoint[checkpoint] = []
            results_by_checkpoint[checkpoint].append(info)
            successful += 1
            print(f"  ✓ {log_file.name}: {checkpoint}, k={info['k_value']}, seed={info['seed']}")
        elif info is None:
            # Try to determine if it was filtered by k-value
            try:
                filename = os.path.basename(str(log_file))
                filename_pattern = r'branch_(\w+)_k([\d\.]+)_seed(\d+)'
                filename_match = re.search(filename_pattern, filename)
                if filename_match:
                    k_val = float(filename_match.group(2))
                    if k_val == int(k_val):
                        k_val = int(k_val)
                    if k_val not in VALID_K_VALUES:
                        filtered += 1
                        print(f"  - {log_file.name}: k={k_val} not in valid k-values, skipped")
                        continue
            except:
                pass
            failed += 1
            print(f"  ✗ {log_file.name}: Could not extract data")
    
    print(f"  Successfully processed: {successful}/{len(log_files)}")
    if filtered > 0:
        print(f"  Filtered (invalid k-values): {filtered}/{len(log_files)}")
    if failed > 0:
        print(f"  Failed: {failed}/{len(log_files)}")
    
    print(f"  Valid k-values: {VALID_K_VALUES}")
    
    # Sort by k_value within each checkpoint
    for checkpoint in results_by_checkpoint:
        results_by_checkpoint[checkpoint].sort(key=lambda x: float(x['k_value']))
    
    return results_by_checkpoint


def main():
    """
    Main function - analyzes logs from both Adam and Muon optimizer directories
    """
    # Log directories at project root
    LOG_DIRS = {
        'adam': 'logs_branch_multi_adam',
        'muon': 'logs_branch_multi_muon'
    }
    
    print("="*70)
    print("Extracting Training Results to CSV")
    print("="*70)
    
    # Create output directory
    os.makedirs('./cbs_results', exist_ok=True)
    
    # Process each optimizer separately
    for optimizer_name, log_dir in LOG_DIRS.items():
        print(f"\n{'='*70}")
        print(f"Processing {optimizer_name.upper()} logs from: {log_dir}")
        print(f"{'='*70}")
        
        # Check if directory exists
        if not os.path.exists(log_dir):
            print(f"WARNING: Directory not found: {log_dir}")
            continue
        
        # Collect results from log files
        results_by_checkpoint = collect_results_from_logs(log_dir)
        
        if not results_by_checkpoint:
            print(f"No valid log files found in {log_dir}!")
            continue
        
        print(f"Found results for {len(results_by_checkpoint)} checkpoints")
        
        # Save raw data to CSV
        csv_file = f'./cbs_results/cbs_data_{optimizer_name}.csv'
        with open(csv_file, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['Checkpoint', 'K-value', 'Seed', 'Train Loss', 'Val Loss', 'Log File'])
            for checkpoint in sorted(results_by_checkpoint.keys()):
                results = sorted(results_by_checkpoint[checkpoint], 
                               key=lambda x: (float(x['k_value']), x['seed']))
                for r in results:
                    writer.writerow([
                        checkpoint,
                        r['k_value'],
                        r['seed'],
                        r['final_train_loss'] if r['final_train_loss'] else '',
                        r['final_val_loss'] if r['final_val_loss'] else '',
                        r['log_file']
                    ])
        
        print(f"\n✓ Saved: {csv_file}")


if __name__ == "__main__":
    main()