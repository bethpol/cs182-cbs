"""
Run the branched training sweep experiment.

This script executes train_sweep.py for each generated config file,
logs outputs to separate files, and collects results.

Usage:
    python run_branch_sweep.py
"""

import os
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing the config files
CONFIG_DIR = "configs_branch_sweep"

# Directory to save logs
LOGS_DIR = "logs_branch_sweep"

# Training script to run
TRAIN_SCRIPT = "../train.py"

# K values (should match generate_branch_configs.py)
K_VALUES = [1, 2, 4, 8, 16, 32, 64]

# =============================================================================


def run_single_branch(k: int, config_path: str, log_path: str) -> dict:
    """
    Run training for a single branch.
    
    Args:
        k: The k-value for this branch
        config_path: Path to the config file
        log_path: Path to save the log output
        
    Returns:
        dict with results (success, duration, etc.)
    """
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Running Branch k={k}")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Log: {log_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Prepare command
    # Using configurator.py pattern: pass config file as argument
    cmd = ["python", TRAIN_SCRIPT, config_path]
    
    # Run training and capture output
    try:
        with open(log_path, 'w') as log_file:
            # Write header to log
            log_file.write(f"{'='*70}\n")
            log_file.write(f"Branch k={k} Training Log\n")
            log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Config: {config_path}\n")
            log_file.write(f"{'='*70}\n\n")
            log_file.flush()
            
            # Run training process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')  # Print to console
                log_file.write(line)  # Write to log file
                log_file.flush()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Write footer to log
            end_time = time.time()
            duration = end_time - start_time
            log_file.write(f"\n{'='*70}\n")
            log_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Duration: {duration:.2f}s ({duration/60:.2f}m)\n")
            log_file.write(f"Exit code: {return_code}\n")
            log_file.write(f"{'='*70}\n")
        
        success = (return_code == 0)
        
        if success:
            print(f"\n✓ Branch k={k} completed successfully in {duration:.2f}s")
        else:
            print(f"\n✗ Branch k={k} failed with exit code {return_code}")
        
        return {
            'k': k,
            'success': success,
            'duration': duration,
            'exit_code': return_code,
            'log_file': log_path
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✗ Branch k={k} failed with exception: {e}")
        
        return {
            'k': k,
            'success': False,
            'duration': duration,
            'error': str(e),
            'log_file': log_path
        }


def collect_results(results: list) -> None:
    """
    Collect and display final results from all branches.
    
    Args:
        results: List of result dictionaries from each branch
    """
    print(f"\n{'='*70}")
    print("Branched Training Sweep - Final Results")
    print(f"{'='*70}\n")
    
    # Create results table
    print(f"{'k':<4} | {'Status':<10} | {'Duration':<15} | {'Log File':<30}")
    print(f"{'-'*4} | {'-'*10} | {'-'*15} | {'-'*30}")
    
    total_duration = 0
    successful_runs = 0
    
    for result in results:
        k = result['k']
        status = "✓ Success" if result['success'] else "✗ Failed"
        duration = result['duration']
        total_duration += duration
        
        if result['success']:
            successful_runs += 1
        
        duration_str = f"{duration:.1f}s ({duration/60:.1f}m)"
        log_file = Path(result['log_file']).name
        
        print(f"{k:<4} | {status:<10} | {duration_str:<15} | {log_file:<30}")
    
    print(f"{'-'*4} | {'-'*10} | {'-'*15} | {'-'*30}")
    print(f"\nSummary:")
    print(f"  Total branches: {len(results)}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {len(results) - successful_runs}")
    print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    
    print(f"\n{'='*70}\n")
    
    # Save results to JSON
    results_file = os.path.join(LOGS_DIR, "sweep_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'successful_runs': successful_runs,
            'total_runs': len(results),
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}\n")


def main():
    """Run the complete branched training sweep."""
    
    print("="*70)
    print("Branched Training Sweep Runner")
    print("="*70)
    print()
    
    # Check if train_sweep.py exists
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Training script not found: {TRAIN_SCRIPT}")
        print("Please ensure train_sweep.py is in the current directory.")
        return
    
    # Check if config directory exists
    if not os.path.exists(CONFIG_DIR):
        print(f"Error: Config directory not found: {CONFIG_DIR}")
        print("Please run generate_branch_configs.py first.")
        return
    
    # Create logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Logs will be saved to: {LOGS_DIR}/")
    print()
    
    # Collect all config files
    config_files = []
    for k in K_VALUES:
        config_filename = f"config_branch_k{k}.py"
        config_path = os.path.join(CONFIG_DIR, config_filename)
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            continue
        
        config_files.append((k, config_path))
    
    if not config_files:
        print("Error: No config files found!")
        return
    
    print(f"Found {len(config_files)} config files")
    print(f"K values: {[k for k, _ in config_files]}")
    print()
    
    # Confirm before starting
    response = input("Start training sweep? [y/N]: ")
    if response.lower() != 'y':
        print("Sweep cancelled.")
        return
    
    print()
    sweep_start_time = time.time()
    
    # Run each branch sequentially
    results = []
    for k, config_path in config_files:
        log_filename = f"branch_k{k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(LOGS_DIR, log_filename)
        
        result = run_single_branch(k, config_path, log_path)
        results.append(result)
        
        # Brief pause between runs
        if k != config_files[-1][0]:  # Not the last one
            print("\nWaiting 5 seconds before next branch...")
            time.sleep(5)
    
    sweep_end_time = time.time()
    sweep_duration = sweep_end_time - sweep_start_time
    
    # Display final results
    collect_results(results)
    
    print(f"Total sweep duration: {sweep_duration:.1f}s ({sweep_duration/60:.1f}m)")
    print("\nAll branches complete!")
    print()
    print("Next steps:")
    print("  1. Review logs in:", LOGS_DIR)
    print("  2. Check training outputs in: out_branch_k*/ directories")
    print("  3. Analyze results on WandB (if enabled)")
    print()


if __name__ == "__main__":
    main()