# model_45m.py
from dataclasses import dataclass
import torch
from model import GPT, GPTConfig

# A ~45M config:
# - vocab_size matches GPT-2 BPE (50304) used in nanoGPT
# - n_layer=8, n_embd=480, n_head=12  -> ~46.5M total params
# (rule-of-thumb: per layer ≈ 12*d^2; token embed ≈ vocab*d)
DEFAULT_45M_CFG = dict(
    block_size = 1024,     # context length
    vocab_size = 50304,    # GPT-2 BPE vocab used in nanoGPT
    n_layer    = 8,
    n_head     = 12,       # 480/12 = 40-dim heads
    n_embd     = 480,
    bias       = False,    # nanoGPT default
    dropout    = 0.0,
)

def build_gpt_45m(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    cfg = GPTConfig(**DEFAULT_45M_CFG)
    model = GPT(cfg)
    model.to(device)
    return model

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = build_gpt_45m()
    total = count_parameters(m)
    print(f"Total parameters: {total:,}")  # expect ~46,000,000
