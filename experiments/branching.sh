#!/bin/bash
#SBATCH --job-name=branch_sweep
#SBATCH --output=slurm_branch_sweep_%j.out
#SBATCH --error=slurm_branch_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# =============================================================================
# SLURM Configuration for Branched Training Sweep
# 
# This script will:
# 1. Generate config files for all k-values
# 2. Run train_sweep.py for each config sequentially
# 3. Save all logs and results
#
# Usage:
#     sbatch run_branch_sweep.slurm
#
# To modify resources, adjust the #SBATCH directives above
# =============================================================================

# Print job information
echo "=========================================="
echo "SLURM Job: Branched Training Sweep"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Load modules (adjust these for your cluster)
# Uncomment and modify as needed for your system
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment if needed
# Uncomment and modify path to your venv
# source /path/to/your/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Print Python and CUDA info
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="slurm_sweep_${TIMESTAMP}.log"

echo "=========================================="
echo "Step 1: Generating Config Files"
echo "=========================================="
echo ""

# Generate configs
python generate_branch_configs.py 2>&1 | tee -a "$MAIN_LOG"

if [ $? -ne 0 ]; then
    echo "ERROR: Config generation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running Training Sweep"
echo "=========================================="
echo ""

# Run the sweep (automatically runs all branches)
# We pipe 'y' to auto-confirm the prompt
echo "y" | python run_branch_sweep.py 2>&1 | tee -a "$MAIN_LOG"

if [ $? -ne 0 ]; then
    echo "ERROR: Training sweep failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
echo "Ended: $(date)"
echo "Main log saved to: $MAIN_LOG"
echo "Individual logs in: logs_branch_sweep/"
echo "Results in: out_branch_k*/"
echo ""

# Optional: Create a summary file
echo "Creating summary..."
SUMMARY_FILE="sweep_summary_${TIMESTAMP}.txt"
{
    echo "Branched Training Sweep Summary"
    echo "================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURM_NODELIST"
    echo "Started: $(date)"
    echo ""
    echo "Output directories created:"
    ls -d out_branch_k* 2>/dev/null || echo "No output directories found"
    echo ""
    echo "Log files created:"
    ls logs_branch_sweep/ 2>/dev/null || echo "No logs directory found"
    echo ""
    echo "For detailed results, see:"
    echo "  - Main log: $MAIN_LOG"
    echo "  - Individual logs: logs_branch_sweep/"
    echo "  - JSON results: logs_branch_sweep/sweep_results.json"
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "Done!"