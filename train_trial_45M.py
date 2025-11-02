# train_trial_45m.py
import math, time, torch
from torch import optim
from model_45M import build_gpt_45m, count_parameters
torch.manual_seed(1337)
import os


# Minimal data loader for char-level tiny shakespeare
# Using the tokenized .bin files created by prepare.py
DATA_DIR = "data/shakespeare_char"
train_bin = os.path.join(DATA_DIR, "train.bin")
val_bin   = os.path.join(DATA_DIR, "val.bin")

assert os.path.exists(train_bin) and os.path.exists(val_bin), "Run data/shakespeare_char/prepare.py first."

import numpy as np
def load_bin(path):
    """Load uint16 token ids from .bin file safely using NumPy."""
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))


def get_batch(split, block_size, batch_size, device):
    data = load_bin(train_bin if split == "train" else val_bin)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    model = build_gpt_45m(device)
    print(f"Params: {count_parameters(model):,}")

    # tiny trial hyperparams
    block_size = 128     # shorter context for speed
    batch_size = 8     # keep small for CPU/low VRAM
    lr         = 3e-4
    steps      = 50    # quick sanity run

    # IMPORTANT: resize position embeddings if needed (we set model cfg at 1024)
    if block_size != model.config.block_size:
        # shrink PE table for the trial; safe as a smoke test
        model.transformer.wpe.weight = torch.nn.Parameter(
            model.transformer.wpe.weight[:block_size].clone()
        )
        model.config.block_size = block_size

    # simple AdamW loop
    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()

    # simple cross-entropy stepper
    for step in range(steps):
        x, y = get_batch("train", block_size, batch_size, device)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step+1) % 20 == 0:
            with torch.no_grad():
                vx, vy = get_batch("val", block_size, batch_size, device)
                vlogits, vloss = model(vx, vy)
            print(f"step {step+1:4d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}")

    # save checkpoint
    os.makedirs("out_trial_45m", exist_ok=True)
    torch.save(model.state_dict(), "out_trial_45m/ckpt.pt")
    print("Saved out_trial_45m/ckpt.pt")

if __name__ == "__main__":
    main()
