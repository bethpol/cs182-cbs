#!/bin/bash

# Redirect all output (stdout and stderr) to a log file
LOG_FILE="logs/nanogpt_test_w_shakespeare_blockSize1024_vocabSize50304_normalAdamw.log"
exec &> "$LOG_FILE"

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate cbs-nanogpt
conda info --envs

# Use absolute python; avoids PATH issues on worker
PY="$CONDA_PREFIX/bin/python"
which "$PY" && "$PY" -V

# Set the WANDB timeout to prevent errors on startup
export WANDB_HTTP_TIMEOUT=60

# Launch
python train.py config/train_gpt_45m_adamw.py

# torchrun --nproc_per_node=2 train_cl_model.py --model resnet50 --dataset lucas_images_x2bin_lowpass --epochs 5 --image_size 300 --exp_name multigpu_cl_big_dataset_changedwayoftransforms_numworkers2commentprefetchfac_2gpu_cassowary_lr_1.4e-3_bs256_removeAllTransformsExceptTensortransform_persistentFalse_noprefetch_simplifiedMainDLoader_removedSVM --batch_size 256 --lr 0.0014 --save_freq 10000 --loss SimCLR --noise2d 0.0 --filter_type lowpass --num_mask_patches_max 10 --fourier_amp_scaling_bins 10
# python train.py \
#       --dataset=shakespeare_char \
#       --max_iters=30 \
#       --lr_decay_iters=30 \
#       --eval_interval=10 \
#       --batch_size=8 \
#       --block_size=128 \
#       --gradient_accumulation_steps=2 \
#       --learning_rate=3e-4