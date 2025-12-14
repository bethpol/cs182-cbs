import json
import os
import matplotlib.pyplot as plt
import shutil
import math

# Define paths relative to the script location
# Structure:
# plots/
#   final_loss_grid.py (this script)
#   final_grid_data/
#     data.json
#   final_loss_grid_plots/
#     (output)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'final_grid_data', 'data.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'final_loss_grid_plots')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def organize_by_setup(data_list):
    # Group by (optimizer, checkpoint)
    # data_list is a list of dicts: {optimizer, checkpoint, k, loss, sem}
    grouped = {}
    for entry in data_list:
        key = (entry['optimizer'], entry['checkpoint'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(entry)
    
    # Sort by k
    for key in grouped:
        grouped[key].sort(key=lambda x: x['k'])
        
    return grouped

def format_title(opt, ckpt):
    opt_map = {"ADAMW": "AdamW", "MUON": "Muon"}
    opt_cased = opt_map.get(opt.upper(), opt.capitalize())
    return f"{opt_cased} {ckpt}"

def get_checkpoint_value(ckpt_str):
    try:
        clean_str = ckpt_str.upper().replace('M', '')
        return float(clean_str)
    except ValueError:
        return 0

def plot_setup_on_ax(ax, opt, ckpt, train_groups, val_groups, cbs_entry, title=None, legend=True):
    key = (opt, ckpt)
    
    # Plot Train
    if key in train_groups:
        entries = train_groups[key]
        ks = [e['k'] for e in entries]
        losses = [e['loss'] for e in entries]
        sems = [e['sem'] for e in entries]
        ax.errorbar(ks, losses, yerr=sems, label='Train Loss', marker='o', capsize=6, linestyle='-', markersize=10, linewidth=3)
        
    # Plot Val
    if key in val_groups:
        entries = val_groups[key]
        ks = [e['k'] for e in entries]
        losses = [e['loss'] for e in entries]
        sems = [e['sem'] for e in entries]
        ax.errorbar(ks, losses, yerr=sems, label='Validation Loss', marker='s', capsize=6, linestyle='--', markersize=10, linewidth=3)
        
    # Add CBS Line and Region
    # cbs_entry is {'mean': X, 'std': Y} or None
    if cbs_entry:
        mean_k = cbs_entry['mean']
        std_k = cbs_entry['std']
        ax.axvline(x=mean_k, color='red', linestyle='--', linewidth=4, label='Mean CBS')
        if std_k > 0:
            ax.axvspan(mean_k - std_k, mean_k + std_k, color='red', alpha=0.15)

    ax.set_xscale('log')
    
    # Styling
    ax.set_xlabel('k (log scale)', fontsize=24)
    ax.set_ylabel('Final Loss', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    full_title = title if title else format_title(opt, ckpt)
    ax.set_title(full_title, fontsize=30)
    
    if legend:
        ax.legend(fontsize=18)
    ax.grid(True, which="both", ls="-", alpha=0.5)

def generate_plots_for_epsilon(epsilon, train_groups, val_groups, cbs_data_for_eps, output_root):
    print(f"Generating plots for Epsilon={epsilon}...")
    
    # Collect all available setups from Training/Val data
    all_setups = set(train_groups.keys()) | set(val_groups.keys())
    
    # --- 1. Main Paper Grid (2x3) ---
    grid_rows = ["ADAMW", "MUON"]
    grid_cols = ["50M", "2000M", "4000M"]
    
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    for i, opt_req in enumerate(grid_rows):
        for j, ckpt_req in enumerate(grid_cols):
            ax = axes[i, j]
            # Key format in JSON cbs is "OPTIMIZER_CHECKPOINT"
            cbs_key = f"{opt_req}_{ckpt_req}"
            cbs_val = cbs_data_for_eps.get(cbs_key)
            
            if (opt_req, ckpt_req) in all_setups:
                plot_setup_on_ax(ax, opt_req, ckpt_req, train_groups, val_groups, cbs_val)
            else:
                ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center', fontsize=24)
                ax.set_title(f"{opt_req} {ckpt_req}", fontsize=30)
                
    plt.tight_layout()
    main_grid_name = f"main_paper_grid_epsilon_{epsilon}.pdf"
    main_grid_path = os.path.join(output_root, main_grid_name)
    plt.savefig(main_grid_path)
    plt.close()
    print(f"Saved: {main_grid_name}")

    # --- 2. Appendix Grid (4 Rows) ---
    all_opt_checkpoints = set([s[1] for s in all_setups])
    sorted_checkpoints = sorted(list(all_opt_checkpoints), key=get_checkpoint_value)
    
    num_checkpoints = len(sorted_checkpoints)
    cols = math.ceil(num_checkpoints / 2) if num_checkpoints > 0 else 1
    rows = 4 # Fixed layout
    
    fig_width = 10 * cols
    fig_height = 10 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if cols == 1: axes = axes.reshape(rows, 1) # Handle single col edge case
    
    for r in range(rows):
        if r < 2:
            opt = "ADAMW"
            start_idx = 0 if r == 0 else cols
        else:
            opt = "MUON"
            start_idx = 0 if r == 2 else cols
            
        for c in range(cols):
            ax = axes[r, c]
            ckpt_idx = start_idx + c
            
            if ckpt_idx < num_checkpoints:
                ckpt = sorted_checkpoints[ckpt_idx]
                cbs_key = f"{opt}_{ckpt}"
                cbs_val = cbs_data_for_eps.get(cbs_key)
                
                if (opt, ckpt) in all_setups:
                    plot_setup_on_ax(ax, opt, ckpt, train_groups, val_groups, cbs_val)
                else:
                    ax.set_axis_off()
            else:
                ax.set_axis_off()
                
    plt.tight_layout()
    app_grid_name = f"appendix_grid_epsilon_{epsilon}.pdf"
    app_grid_path = os.path.join(output_root, app_grid_name)
    plt.savefig(app_grid_path)
    plt.close()
    print(f"Saved: {app_grid_name}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please ensure it exists.")
        return

    data = load_data()
    train_groups = organize_by_setup(data['train_loss'])
    val_groups = organize_by_setup(data['val_loss'])
    
    ensure_dir(OUTPUT_DIR)
    
    # Iterate over all epsilons in CBS data
    # cbs data structure: { "0.5": { "ADAMW_50M": {...} }, ... }
    for epsilon, cbs_vals in data['cbs'].items():
        generate_plots_for_epsilon(epsilon, train_groups, val_groups, cbs_vals, OUTPUT_DIR)

if __name__ == "__main__":
    main()
