# Optimizer Customization Guide

## Overview

This version of the training script `train_sweep.py` can be used to run hyperparameter search experiments on a single gpu. Currently this version of the code doesn't support training on multiple gpus.

## Additional files for this experiment:

- **train_sweep.py**: main train code
- **optimizers.py**: define your favorite optimizer here which can then be loaded in the main train_sweep.py code
- Configuration file: eg: **train_sweep_muon_configs.py**

### Steps to run this code:

### 1. First create a configuration file and define your configs

* **out_dir**: Output directory to where all checkpoints will be saved. The checkpoints in this directory will have the following format in the filename eg: tokens(3.07e3)_tloss(10.910)_vloss(10.924)_ckpt.pt
* **init_from**:
    - 'scratch': a fresh training run
    - 'resume': if you want to resume from a checkpoint. if yes you also have to define the resume_ckpt_fname configuration and adding the checkpoint file name to start the training run from
* **log_interval_tokens**: The wandb/ terminal logging frequency in tokens(seen)
* **checkpoint_interval_tokens**: Checkpoint saving frequency in tokens(seen). Additionally a final checkpoint will always be saved at the end of a complete training run.
* **eval_iters**: The number of iterations/ steps you want to take through the validation set to record and monitor the validation loss
* **wandb_run_name**: The experiment name that should be logged on wandb
* **checkpoint_token_pos**: Should be set to 0 if it's a fresh run. Else if init_from is set to 'resume' this will be determined and overwritten internally from the checkpoint.
* **batch_size**: Training batch size
* **gradient_accumulation_steps**: If you want a larger batchsize but can't fit everything in vram, then you can set this to a >1 which will then run a loop to accumulate batches. The effective batch size = batch_size * gradient_accumulation_steps (without ddp version)
* **max_tokens**: The maximum number of tokens to run through the dataset for one full training run.
* **optimizer_type**:
    - 'adamw': AdamW optimizer
    - 'muon': Muon optimizer
* **learning_rate**: Learning rate to set for the optimizer. In the current version there's no scheduler to update the learning rate.
* **weight_decay**: Weight decay parameter to set for the optimizers

### 2. Run the script once the configs are defined

```bash
python train_sweep.py config/train_sweep_adam_configs.py
```