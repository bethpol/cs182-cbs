# Critical Batch Size Study (nanoGPT-45M)

This repository contains the **reproducible implementation** for our Critical Batch Size (CBS) study on modern optimizers (AdamW vs Muon) using a custom **45 million-parameter nanoGPT** model.

This repo provides a **minimal, importable nanoGPT core** that supports:
- Lightweight 45M model (`model_45m.py`)
- Shared dataset and loader setup
- Branching experiment loop (CBS measurement)
- Large-scale training and checkpointing loop

It is collaboratively developed by:
- **Sammie Smith** – Model setup (45 M, training smoke test)
- **Liz Weaver** – Dataset preparation (random subset, 20B-token limit)
- **Beth Polito** – Branching experiment logic
- **Kithmini Herath** – Large training + checkpointing system

---

## Repository Structure
```bash
cbs-nanogpt-study/
│
├── model_45m.py # Sammie: defines 45M-param model (importable)
├── train_trial_45m.py # Sammie: quick smoke test for training
│
├── model.py # core GPT architecture (from nanoGPT)
├── train.py # nanoGPT training logic (for large runs)
├── configurator.py # loads config/*.py training configs
│
├── data/
│ ├── shakespeare_char/prepare.py # example dataset prep script
│ ├── loaders/ # Liz: random subset + dataloader utilities
│ └── init.py
│
├── experiments/
│ ├── branching_experiment.py # Beth: CBS measurement loop
│ ├── train_large_loop.py # Kithmini: full training & checkpointing
│ └── configs/
│ ├── adamw_45m.yaml
│ ├── muon_45m.yaml
│ └── ...
│
├── utils/
│ ├── param_counter.py
│ ├── seed_utils.py
│ └── init.py
│
├── requirements.txt # dependencies (for pip)
└── README.md # this file
```



---

## Environment Setup (Conda)

> We recommend using **Python 3.10** with a dedicated conda environment.

```bash
# 1) Create and activate the environment
conda create -n cbs-nanogpt python=3.10 -y
conda activate cbs-nanogpt

# 2) Install PyTorch 2.9
# Select the installation option based on your CUDA version on Linux/Windows systems:
# a) CUDA 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# or
# b) CUDA 12.8
pip3 install torch torchvision
# or
# c) CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# ---- OR CPU-only build (for Linux) ----
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) Install project dependencies
pip install numpy tqdm tiktoken datasets wandb matplotlib
```

## Verify Installation

Clone the repository and run a quick smoke test

```bash
git clone https://github.com/bethpol/cs182-cbs.git
cd cs182-cbs
python train_trial_45m.py
```
You should see an output similar to:

```bash
Params: 46,764,000
step   20 | train loss 3.25 | val loss 3.29
step   40 | train loss 3.18 | val loss 3.25
...
Saved out_trial_45m/ckpt.pt
```

This confirms that the model builds correctly (~45M parameters), training and validation losses decrease, and dependencies are working across OS/PyTorch versions


## Trial Training

To verify that the model trains end to end, use a timy shakespeare char dataset
```bash
# Prepare data
python data/shakespeare_char/prepare.py # this creates train.bin, val.bin. I have already generated these for convenience, so you can skip this step.

# Run quick training (≈1–3 min on GPU, 10–15 min on CPU)
python train_trial_45m.py
```


## Model Overview (model_45M.py)
```bash
from model import GPT, GPTConfig

DEFAULT_45M_CFG = dict(
    block_size=1024,
    vocab_size=50304,
    n_layer=8,
    n_head=12,
    n_embd=480,
    bias=False,
    dropout=0.0,
)

def build_gpt_45m(device="cuda"):
    cfg = GPTConfig(**DEFAULT_45M_CFG)
    model = GPT(cfg)
    return model.to(device)
```
To import and use in any script:

```bash
from model_45m import build_gpt_45m
model = build_gpt_45m(device="cuda")
```


