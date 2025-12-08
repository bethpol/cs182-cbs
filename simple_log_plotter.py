"""
Simple log file plotter - reads local log files and makes plots.
No W&B API, no timeouts, just works.

Usage:
    python simple_log_plotter.py
"""

import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories containing your log files
LOG_DIRS = {
    'adam': 'logs_branch_multi_adam',
    'muon': 'logs_branch_multi_muon'
}

# Log filename pattern: branch_{checkpoint}_k{k}_seed{seed}_{timestamp}.log
LOG_PATTERN = 'branch_*.log'

# Output directory
OUTPUT_DIR = 'plots_new'

# Number of points to interpolate to (makes all curves have same frequency)
N_INTERPOLATION_POINTS = 50

# Plot styling
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0)

# Font sizes for different elements
TITLE_FONTSIZE = 30
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 25
LEGEND_FONTSIZE = 25

# Valid k-values to include (same as CBS analysis)
VALID_K_VALUES = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]

# ============================================================================
# COLOR MAPPING - Consistent colors for k values across all plots
# ============================================================================

def get_color_map_for_k_values(k_values):
    """
    Create a consistent color mapping for k values.
    Same k value will always get the same color across all plots.
    """
    k_values_sorted = sorted(k_values)
    colors = sns.color_palette("husl", len(k_values_sorted))
    return dict(zip(k_values_sorted, colors))

# Global color map (will be initialized after loading data)
K_VALUE_COLOR_MAP = {}

# ============================================================================
# PARSER
# ============================================================================

def parse_filename(filename):
    """Extract checkpoint, k, seed from filename."""
    # k can be integer `8` or float `0.5`
    pattern = r'branch_(\d+M)_k([\d\.]+)_seed(\d+)_'
    match = re.search(pattern, filename)
    if match:
        checkpoint = match.group(1)
        k_value = float(match.group(2))
        seed = int(match.group(3))
        return checkpoint, k_value, seed
    return None, None, None


def parse_log_file(filepath):
    """Parse training/validation loss from log file."""
    # Try UTF-8, fall back to latin-1 for special characters
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Parse training lines: tokens: 507,904 | loss: 4.0437 | grad_norm: 0.9287
    train_pattern = r'tokens:\s+([\d,]+)\s+\|\s+loss:\s+([\d.]+)\s+\|\s+grad_norm:\s+([\d.]+)'
    train_matches = re.findall(train_pattern, content)
    
    train_data = []
    for tokens_str, loss_str, grad_norm_str in train_matches:
        tokens = int(tokens_str.replace(',', ''))
        loss = float(loss_str)
        train_data.append({'tokens': tokens, 'train_loss': loss})
    
    # Parse validation lines
    val_pattern = r'tokens:\s+([\d,]+).*?\nValidation loss:\s+([\d.]+)'
    val_matches = re.findall(val_pattern, content, re.MULTILINE)
    
    val_data = []
    for tokens_str, val_loss_str in val_matches:
        tokens = int(tokens_str.replace(',', ''))
        val_loss = float(val_loss_str)
        val_data.append({'tokens': tokens, 'val_loss': val_loss})
    
    return train_data, val_data


def load_all_logs(log_dirs):
    """Load all log files from multiple directories (adam and muon)."""
    all_data = []
    
    for optimizer, log_dir in log_dirs.items():
        log_dir_path = Path(log_dir)
        
        if not log_dir_path.exists():
            print(f"⚠️  Directory not found: {log_dir}, skipping {optimizer}")
            continue
        
        log_files = list(log_dir_path.glob(LOG_PATTERN))
        
        if not log_files:
            print(f"⚠️  No log files found in {log_dir}")
            continue
        
        print(f"\n📁 Loading {optimizer.upper()} logs from {log_dir}/...")
        print(f"   Found {len(log_files)} log files")
        
        for log_file in log_files:
            checkpoint, k_value, seed = parse_filename(log_file.name)
            
            if checkpoint is None:
                print(f"  ⚠️  Skipping {log_file.name} (doesn't match pattern)")
                continue
            
            # Filter by valid k-values
            if k_value not in VALID_K_VALUES:
                print(f"  - {log_file.name}: k={k_value} not in valid k-values, skipped")
                continue
            
            print(f"  ✓ {log_file.name}")
            
            try:
                train_data, val_data = parse_log_file(log_file)
                
                # Convert to DataFrame
                if train_data:
                    df_train = pd.DataFrame(train_data)
                    df_train['checkpoint'] = checkpoint
                    df_train['k_value'] = k_value
                    df_train['seed'] = seed
                    df_train['metric'] = 'train_loss'
                    df_train['optimizer'] = optimizer
                    all_data.append(df_train)
                
                if val_data:
                    df_val = pd.DataFrame(val_data)
                    df_val['checkpoint'] = checkpoint
                    df_val['k_value'] = k_value
                    df_val['seed'] = seed
                    df_val['metric'] = 'val_loss'
                    df_val['optimizer'] = optimizer
                    all_data.append(df_val)
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
    
    if not all_data:
        raise ValueError("No data extracted from logs")
    
    # Combine everything
    df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✓ Loaded {len(df)} raw data points")
    print(f"  Optimizers: {sorted(df['optimizer'].unique())}")
    print(f"  Checkpoints: {sorted(df['checkpoint'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  K values: {sorted(df['k_value'].unique())}")
    print(f"  Valid k-values filter: {VALID_K_VALUES}")
    
    return df


def interpolate_to_common_points(df, n_points=N_INTERPOLATION_POINTS):
    """
    Interpolate all runs to have the same token positions.
    This ensures consistent frequency across all checkpoints and makes averaging meaningful.
    """
    print(f"\n🔄 Interpolating all runs to {n_points} common points...")
    
    interpolated_data = []
    
    # Group by unique run (checkpoint, k_value, seed, metric, optimizer)
    for (checkpoint, k_value, seed, metric, optimizer), group in df.groupby(
        ['checkpoint', 'k_value', 'seed', 'metric', 'optimizer']
    ):
        group = group.sort_values('tokens')
        
        if len(group) < 2:
            print(f"  ⚠️  Skipping {checkpoint}, k={k_value}, seed={seed}, {metric}, {optimizer} (insufficient data)")
            continue
        
        # Get token range for this run
        tokens = group['tokens'].values
        token_min = tokens.min()
        token_max = tokens.max()
        
        # Get loss values
        if metric == 'train_loss':
            loss_values = group['train_loss'].values
        else:
            loss_values = group['val_loss'].values
        
        # Create common token positions (evenly spaced)
        common_tokens = np.linspace(token_min, token_max, n_points)
        
        # Interpolate loss values to common token positions
        try:
            f = interp1d(tokens, loss_values, kind='linear', fill_value='extrapolate')
            interpolated_loss = f(common_tokens)
            
            # Create new dataframe with interpolated values
            for tok, loss in zip(common_tokens, interpolated_loss):
                interpolated_data.append({
                    'tokens': tok,
                    metric: loss,
                    'checkpoint': checkpoint,
                    'k_value': k_value,
                    'seed': seed,
                    'metric': metric,
                    'optimizer': optimizer
                })
        except Exception as e:
            print(f"  ⚠️  Interpolation failed for {checkpoint}, k={k_value}, seed={seed}, {metric}: {e}")
            continue
    
    df_interpolated = pd.DataFrame(interpolated_data)
    
    print(f"✓ Interpolated to {len(df_interpolated)} data points")
    print(f"  Each run now has exactly {n_points} evenly-spaced points")
    
    return df_interpolated


def plot_all_k_on_same_axis(df, optimizer, checkpoint, metric_name, output_dir):
    """
    Plot all k values on the same axis for a specific optimizer and checkpoint.
    Shows mean ± SEM for each k value.
    """
    df_plot = df[(df['optimizer'] == optimizer) &
                 (df['checkpoint'] == checkpoint) & 
                 (df['metric'] == metric_name)].copy()
    
    if df_plot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metric_col = metric_name
    k_values = sorted(df_plot['k_value'].unique())
    
    for k_val in k_values:
        df_k = df_plot[df_plot['k_value'] == k_val]
        
        # Aggregate across seeds
        agg = df_k.groupby('tokens')[metric_col].agg([
            ('mean', 'mean'),
            ('sem', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0)
        ]).reset_index()
        
        tokens = agg['tokens'].values / 1e6
        mean = agg['mean'].values
        sem = agg['sem'].values
        
        color = K_VALUE_COLOR_MAP[k_val]
        ax.plot(tokens, mean, linewidth=2.5, color=color, label=f'k={k_val}', alpha=0.9)
        ax.fill_between(tokens, mean - sem, mean + sem, alpha=0.2, color=color)
    
    ax.set_xlabel('Tokens Seen (Millions)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=LABEL_FONTSIZE)
    if optimizer.upper() == 'ADAM':
        optimizer_title = 'ADAMW'
    else:
        optimizer_title = 'MUON'

    ax.set_title(f'{optimizer_title} - {checkpoint} - All K Values - {metric_name.replace("_", " ").title()} (Mean ± SEM)', 
                fontsize=TITLE_FONTSIZE, pad=20)
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    filename = f'{output_dir}/{optimizer}_{checkpoint}_all_k_same_axis_{metric_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ {filename}")
    plt.close()

def plot_all_k_single_seeds(df, optimizer, checkpoint, metric_name, output_dir):
    """
    Plot all k values for each individual seed on separate subplots.
    Creates one figure with 3 subplots (one per seed).
    """
    df_plot = df[(df['optimizer'] == optimizer) &
                 (df['checkpoint'] == checkpoint) & 
                 (df['metric'] == metric_name)].copy()
    
    if df_plot.empty:
        return
    
    seeds = sorted(df_plot['seed'].unique())
    n_seeds = len(seeds)
    
    if n_seeds == 0:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_seeds, figsize=(12 * n_seeds, 7), squeeze=False)
    axes = axes.flatten()
    
    metric_col = metric_name
    k_values = sorted(df_plot['k_value'].unique())
    
    for idx, seed in enumerate(seeds):
        ax = axes[idx]
        df_seed = df_plot[df_plot['seed'] == seed]
        
        for k_val in k_values:
            df_k = df_seed[df_seed['k_value'] == k_val]
            
            if df_k.empty:
                continue
            
            # Get data for this k value and seed
            df_k_sorted = df_k.sort_values('tokens')
            tokens = df_k_sorted['tokens'].values / 1e6
            loss = df_k_sorted[metric_col].values
            
            color = K_VALUE_COLOR_MAP[k_val]
            ax.plot(tokens, loss, linewidth=2.5, color=color, label=f'k={k_val}', alpha=0.9)
        
        ax.set_xlabel('Tokens Seen (Millions)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=LABEL_FONTSIZE)
        
        if optimizer.upper() == 'ADAM':
            optimizer_title = 'ADAMW'
        else:
            optimizer_title = 'MUON'
        
        ax.set_title(f'{optimizer_title} - {checkpoint} - Seed {seed}', 
                    fontsize=TITLE_FONTSIZE, pad=20)
        ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    filename = f'{output_dir}/{optimizer}_{checkpoint}_individual_seeds_{metric_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ {filename}")
    plt.close()

def generate_validation_table(df, checkpoints, k_values, seeds, optimizers):
    """
    Generate a table with validation loss results for both optimizers.
    Shows final validation loss (mean ± SEM), min, and max for each checkpoint, optimizer, and k value.
    """
    # Filter for validation loss only
    df_val = df[df['metric'] == 'val_loss'].copy()
    
    table_data = []
    
    for optimizer in optimizers:
        for checkpoint in checkpoints:
            for k_val in k_values:
                df_subset = df_val[(df_val['optimizer'] == optimizer) &
                                  (df_val['checkpoint'] == checkpoint) & 
                                  (df_val['k_value'] == k_val)]
                
                if df_subset.empty:
                    continue
                
                # Get final token position (last 10% of data to average over end)
                max_tokens = df_subset['tokens'].max()
                final_threshold = max_tokens * 0.9  # Last 10% of training
                
                df_final = df_subset[df_subset['tokens'] >= final_threshold]
                
                if df_final.empty:
                    df_final = df_subset  # Fallback to all data
                
                # Calculate mean and SEM across seeds
                final_losses = df_final.groupby('seed')['val_loss'].mean()
                
                mean_loss = final_losses.mean()
                sem_loss = final_losses.std() / np.sqrt(len(final_losses)) if len(final_losses) > 1 else 0
                n_seeds = len(final_losses)
                
                # Get min and max for reference
                min_loss = df_subset['val_loss'].min()
                max_loss = df_subset['val_loss'].max()
                
                table_data.append({
                    'Optimizer': optimizer.upper(),
                    'Checkpoint': checkpoint,
                    'K': k_val,
                    'Final Val Loss (Mean)': f'{mean_loss:.4f}',
                    'SEM': f'{sem_loss:.4f}',
                    'Mean ± SEM': f'{mean_loss:.4f} ± {sem_loss:.4f}',
                    'Min (any point)': f'{min_loss:.4f}',
                    'Max (any point)': f'{max_loss:.4f}',
                    'N Seeds': n_seeds
                })
    
    df_table = pd.DataFrame(table_data)
    
    return df_table


def generate_training_loss_table(df, checkpoints, k_values, seeds, optimizers):
    """
    Generate a table with training loss results for both optimizers.
    Shows final training loss (mean ± SEM), min, and max for each checkpoint, optimizer, and k value.
    """
    # Filter for training loss only
    df_train = df[df['metric'] == 'train_loss'].copy()
    
    table_data = []
    
    for optimizer in optimizers:
        for checkpoint in checkpoints:
            for k_val in k_values:
                df_subset = df_train[(df_train['optimizer'] == optimizer) &
                                    (df_train['checkpoint'] == checkpoint) & 
                                    (df_train['k_value'] == k_val)]
                
                if df_subset.empty:
                    continue
                
                # Get final token position (last 10% of data to average over end)
                max_tokens = df_subset['tokens'].max()
                final_threshold = max_tokens * 0.9  # Last 10% of training
                
                df_final = df_subset[df_subset['tokens'] >= final_threshold]
                
                if df_final.empty:
                    df_final = df_subset  # Fallback to all data
                
                # Calculate mean and SEM across seeds
                final_losses = df_final.groupby('seed')['train_loss'].mean()
                
                mean_loss = final_losses.mean()
                sem_loss = final_losses.std() / np.sqrt(len(final_losses)) if len(final_losses) > 1 else 0
                n_seeds = len(final_losses)
                
                # Get min and max for reference
                min_loss = df_subset['train_loss'].min()
                max_loss = df_subset['train_loss'].max()
                
                table_data.append({
                    'Optimizer': optimizer.upper(),
                    'Checkpoint': checkpoint,
                    'K': k_val,
                    'Final Train Loss (Mean)': f'{mean_loss:.4f}',
                    'SEM': f'{sem_loss:.4f}',
                    'Mean ± SEM': f'{mean_loss:.4f} ± {sem_loss:.4f}',
                    'Min (any point)': f'{min_loss:.4f}',
                    'Max (any point)': f'{max_loss:.4f}',
                    'N Seeds': n_seeds
                })
    
    df_table = pd.DataFrame(table_data)
    
    return df_table

def generate_latex_tables(train_table, val_table, output_dir):
    """
    Generate LaTeX formatted tables from the training and validation loss tables.
    Excludes the N Seeds column.
    """
    latex_content = []
    
    # Helper function to convert dataframe to LaTeX
    def df_to_latex(df, caption, label):
        # Remove N Seeds column
        df_latex = df.drop(columns=['N Seeds'], errors='ignore')

        # Replace ± with $\pm$ in the 'Mean ± SEM' column
        if 'Mean ± SEM' in df_latex.columns:
            df_latex['Mean ± SEM'] = df_latex['Mean ± SEM'].str.replace('±', r'$\pm$')
        
        
        # Convert to LaTeX
        latex = df_latex.to_latex(
            index=False,
            caption=caption,
            label=label,
            column_format='l' * len(df_latex.columns),
            escape=False,
            float_format='%.4f'
        )
        return latex
    
    # Training Loss Table
    latex_content.append("% Training Loss Table")
    latex_content.append(df_to_latex(
        train_table,
        caption="Training Loss Results (Final Values - Mean ± SEM)",
        label="tab:training_loss"
    ))
    latex_content.append("\n")
    
    # Validation Loss Table
    latex_content.append("% Validation Loss Table")
    latex_content.append(df_to_latex(
        val_table,
        caption="Validation Loss Results (Final Values - Mean ± SEM)",
        label="tab:validation_loss"
    ))
    
    # Write to file
    latex_file = f'{output_dir}/loss_tables.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"💾 Saved: {latex_file}")
    
    return latex_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    global K_VALUE_COLOR_MAP
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all logs
    print(f"📂 Loading logs from {LOG_DIRS}/...")
    df_raw = load_all_logs(LOG_DIRS)
    
    # Initialize global color map for k values
    all_k_values = sorted(df_raw['k_value'].unique())
    K_VALUE_COLOR_MAP = get_color_map_for_k_values(all_k_values)
    print(f"\n🎨 Color mapping for k values:")
    for k_val, color in K_VALUE_COLOR_MAP.items():
        print(f"  k={k_val}: RGB{tuple(np.round(np.array(color) * 255).astype(int))}")
    
    # Save raw data
    csv_file = f'{OUTPUT_DIR}/raw_data.csv'
    df_raw.to_csv(csv_file, index=False)
    print(f"\n💾 Saved: {csv_file}")
    
    # Interpolate all runs to common token positions
    df = interpolate_to_common_points(df_raw, n_points=N_INTERPOLATION_POINTS)
    
    # Save interpolated data
    csv_file_interp = f'{OUTPUT_DIR}/interpolated_data.csv'
    df.to_csv(csv_file_interp, index=False)
    print(f"💾 Saved: {csv_file_interp}")
    
    # Get unique values
    optimizers = sorted(df['optimizer'].unique())
    checkpoints = sorted(df['checkpoint'].unique(), key=lambda x: int(x.replace('M', '')))
    k_values = sorted(df['k_value'].unique())
    seeds = sorted(df['seed'].unique())
    
    # PLOT TYPE: All k values on same axis, for each optimizer and checkpoint (TRAIN ONLY)
    print(f"\n📊 Creating plots: All k values on same axis per optimizer/checkpoint (train loss only)...")
    for optimizer in optimizers:
        for checkpoint in checkpoints:
            plot_all_k_on_same_axis(df, optimizer, checkpoint, 'train_loss', OUTPUT_DIR)

    # PLOT TYPE: Individual seeds for 4000M AdamW only
    print(f"\n📊 Creating individual seed plots for AdamW 4000M...")
    plot_all_k_single_seeds(df, 'adam', '4000M', 'train_loss', OUTPUT_DIR)
    
    # VALIDATION LOSS TABLE
    print(f"\n📋 Generating validation loss table...")
    val_table = generate_validation_table(df, checkpoints, k_values, seeds, optimizers)
    
    # Save table to CSV
    table_file = f'{OUTPUT_DIR}/validation_loss_table.csv'
    val_table.to_csv(table_file, index=False)
    print(f"💾 Saved: {table_file}")
    
    # TRAINING LOSS TABLE
    print(f"\n📋 Generating training loss table...")
    train_table = generate_training_loss_table(df, checkpoints, k_values, seeds, optimizers)
    
    # Save table to CSV
    train_table_file = f'{OUTPUT_DIR}/training_loss_table.csv'
    train_table.to_csv(train_table_file, index=False)
    print(f"💾 Saved: {train_table_file}")

    # Generate LaTeX tables
    print(f"\n📄 Generating LaTeX tables...")
    generate_latex_tables(train_table, val_table, OUTPUT_DIR)
    
    # Display tables
    print(f"\n" + "="*80)
    print("VALIDATION LOSS RESULTS (Final Values - Mean ± SEM)")
    print("="*80)
    print(val_table.to_string(index=False))
    print("="*80)
    
    print(f"\n" + "="*80)
    print("TRAINING LOSS RESULTS (Final Values - Mean ± SEM)")
    print("="*80)
    print(train_table.to_string(index=False))
    print("="*80)
    
    print(f"\n✅ Done! All plots and tables in {OUTPUT_DIR}/")
    print(f"\nGenerated:")
    print(f"  - {len(optimizers) * len(checkpoints)} plots (all k on same axis per optimizer - train)")
    print(f"  - 1 validation loss table (CSV + displayed above)")
    print(f"  - 1 training loss table (CSV + displayed above)")
    print(f"\nNote: All plots now use {N_INTERPOLATION_POINTS} interpolated points for consistent frequency")
    print(f"Note: All k values use consistent colors across all plots")
    print(f"Note: Only k-values in {VALID_K_VALUES} are included")


if __name__ == "__main__":
    main()