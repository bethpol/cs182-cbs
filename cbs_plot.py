import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate CBS for each checkpoint
def calculate_cbs(df, checkpoint, epsilon=0.5):
    # Filter data for this checkpoint
    ckpt_data = df[df['Checkpoint'] == checkpoint]
    
    # Group by K-value and calculate mean validation loss across seeds
    k_losses = ckpt_data.groupby('K-value')['Val Loss'].agg(['mean', 'std']).reset_index()
    k_losses.columns = ['K-value', 'mean_loss', 'std_loss']
    
    # Find minimum loss
    min_loss = k_losses['mean_loss'].min()
    
    # Find largest K where loss <= min_loss + epsilon
    valid_k = k_losses[k_losses['mean_loss'] <= min_loss + epsilon]['K-value']
    
    if len(valid_k) > 0:
        cbs = valid_k.max()
        return cbs
    return None

# Function to get CBS for each seed individually
def get_cbs_by_seed(df, checkpoint, epsilon=0.5):
    ckpt_data = df[df['Checkpoint'] == checkpoint]
    seeds = ckpt_data['Seed'].unique()
    
    results = {}
    for seed in seeds:
        seed_data = ckpt_data[ckpt_data['Seed'] == seed]
        min_loss = seed_data['Val Loss'].min()
        valid_k = seed_data[seed_data['Val Loss'] <= min_loss + epsilon]['K-value']
        
        if len(valid_k) > 0:
            results[seed] = valid_k.max()
        else:
            results[seed] = None
    
    return results

# Read both datasets
df_adam = pd.read_csv('./cbs_results/cbs_data_adam.csv')
df_muon = pd.read_csv('./cbs_results/cbs_data_muon.csv')

# Calculate CBS for each checkpoint
checkpoints = ['50M', '500M', '1000M', '1500M', '2000M', '3000M', '4000M']
tokens = [50e6, 500e6, 1e9, 1.5e9, 2e9, 3e9, 4e9]

# ADAM
adam_cbs = []
adam_errors = []
adam_by_seed = {}
for ckpt in checkpoints:
    seed_cbs = get_cbs_by_seed(df_adam, ckpt)
    adam_by_seed[ckpt] = seed_cbs
    
    # Calculate mean and std of CBS across seeds
    cbs_values = [v for v in seed_cbs.values() if v is not None]
    if cbs_values:
        adam_cbs.append(np.mean(cbs_values))
        adam_errors.append(np.std(cbs_values)/ np.sqrt(3) * 1.96)
    else:
        adam_cbs.append(np.nan)
        adam_errors.append(0)

# MUON
muon_cbs = []
muon_errors = []
muon_by_seed = {}
for ckpt in checkpoints:
    seed_cbs = get_cbs_by_seed(df_muon, ckpt)
    muon_by_seed[ckpt] = seed_cbs
    
    # Calculate mean and std of CBS across seeds
    cbs_values = [v for v in seed_cbs.values() if v is not None]
    if cbs_values:
        muon_cbs.append(np.mean(cbs_values))
        muon_errors.append(np.std(cbs_values)/ np.sqrt(3) * 1.96)
    else:
        muon_cbs.append(np.nan)
        muon_errors.append(0)

# Print detailed table
print("\n" + "="*80)
print("CRITICAL BATCH SIZE (CBS) BY CHECKPOINT, SEED, AND OPTIMIZER")
print("="*80)
print(f"{'Checkpoint':<12} {'Optimizer':<10} {'Seed 0':<10} {'Seed 1':<10} {'Seed 2':<10} {'Mean CBS':<12} {'Std CBS':<12}")
print("-"*80)

for ckpt, adam_mean, adam_std, muon_mean, muon_std in zip(checkpoints, adam_cbs, adam_errors, muon_cbs, muon_errors):
    # ADAM row
    adam_seeds = adam_by_seed[ckpt]
    adam_s0 = adam_seeds.get(0, 'N/A')
    adam_s1 = adam_seeds.get(1, 'N/A')
    adam_s2 = adam_seeds.get(2, 'N/A')
    print(f"{ckpt:<12} {'ADAM':<10} {str(adam_s0):<10} {str(adam_s1):<10} {str(adam_s2):<10} {adam_mean:<12.0f} {adam_std:<12.2f}")
    
    # MUON row
    muon_seeds = muon_by_seed[ckpt]
    muon_s0 = muon_seeds.get(0, 'N/A')
    muon_s1 = muon_seeds.get(1, 'N/A')
    muon_s2 = muon_seeds.get(2, 'N/A')
    print(f"{ckpt:<12} {'MUON':<10} {str(muon_s0):<10} {str(muon_s1):<10} {str(muon_s2):<10} {muon_mean:<12.0f} {muon_std:<12.2f}")
    print("-"*80)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nADAM CBS values by checkpoint:")
for ckpt, cbs, err in zip(checkpoints, adam_cbs, adam_errors):
    print(f"{ckpt}: {cbs:.0f} ± {err:.2f}")

print("\nMUON CBS values by checkpoint:")
for ckpt, cbs, err in zip(checkpoints, muon_cbs, muon_errors):
    print(f"{ckpt}: {cbs:.0f} ± {err:.2f}")

# Plot
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 6))

# Plot ADAM (dark blue)
plt.errorbar(tokens, adam_cbs, yerr=adam_errors,
             marker='o', markersize=8, linewidth=3,
             color="#0B3C78", capsize=5, capthick=2,
             label="ADAMW (Eff. CBS)")

# Plot MUON (light blue)
plt.errorbar(tokens, muon_cbs, yerr=muon_errors,
             marker='s', markersize=8, linewidth=3,
             color="#4FA3FF", capsize=5, capthick=2,
             label="MUON (Eff. CBS)")

plt.yscale("log")
plt.xticks(tokens, checkpoints)
plt.xlabel("Tokens Seen (Checkpoint)")
plt.ylabel("Effective Critical Batch Size")
plt.title("Effective CBS vs. Training Progress")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("effective_cbs_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()