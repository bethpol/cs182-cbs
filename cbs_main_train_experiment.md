# Large training run with checkpointing guide

## Overview

The training script `train.py` can be used to run hyperparameter search experiments on a single gpu. This version of the code doesn't support training on multiple gpus.

## Additional files for this experiment:

- **train.py**: main train code
- Configuration file: eg: **train_large_muon_scheduler_configs.py**

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
* **lr_muon_adam**: Learning rate to use with the additional AdamW optimier. ONLY needs to be set if Muon is used as an optimizer.
* **wd_muon_adam**: Weight decay to use with the additional AdamW optimier. ONLY needs to be set if Muon is used as an optimizer.
* **use_scheduler**: Set to True to use a learning rate scheduler. Current scheduler has 3 phases:
    1. learning rate warm up phase
    2. stable phase (optional)
    3. learning rate decay phase according to a cosine schedule
* **warmup_tokens**: Number of tokens to be set for the learning rate warm up phase
* **stable_tokens**: Number of tokens to be set for the learning rate stable phase. __This should be set to 0 if you want to ignore the stable phase__
* **min_lr_factor**: The minimum learning rate factor ranging within [0,1]. This would be multiplied by the peak learning rate (i.e. learning rate defined by the user for the optimizer above) to determine the minimum learning rate by the scheduler. 

### 2. Run the script once the configs are defined

```bash
python train.py config/train_large_muon_scheduler_configs.py
```