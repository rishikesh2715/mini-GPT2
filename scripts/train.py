#!/usr/bin/env python3
"""
train.py — iterative‐based training for Mini‑GPT‑2 (no DDP),
loading hyperparameters from train_config.yaml,
with Weights & Biases logging for loss & perplexity.
"""

import os
import sys
import time
import math
import pickle
import glob
from contextlib import nullcontext

import yaml
import numpy as np
import torch
from torch.amp import GradScaler
import wandb
from tqdm import trange

from model import GPTConfig, GPT

# -------- Performance Flags --------
torch.backends.cudnn.benchmark = True        # fastest CUDNN kernels for static shapes
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Load hyperparameters from YAML
# -----------------------------------------------------------------------------
with open("config/train_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Cast numeric strings to floats if necessary
learning_rate = float(cfg["learning_rate"])
min_lr        = float(cfg["min_lr"])
weight_decay  = float(cfg["weight_decay"])

# Required params
block_size      = int(cfg["block_size"])
vocab_size      = int(cfg.get("vocab_size", 50304))
n_layer         = int(cfg["n_layer"])
n_head          = int(cfg["n_head"])
n_embd          = int(cfg["n_embd"])
dropout         = float(cfg["dropout"])
batch_size      = int(cfg["batch_size"])

# Iteration‐based scheduling params
warmup_iters               = int(cfg.get("warmup_iters", cfg.get("warmup_epochs", 5)))
max_iters                  = int(cfg.get("max_iters", 5000))
lr_decay_iters             = max_iters

# Fixed‑in‑script control params
eval_interval              = int(cfg.get("eval_interval", 250))
log_interval               = int(cfg.get("log_interval", 10))
eval_iters                 = int(cfg.get("eval_iters", 200))
always_save_checkpoint     = True
gradient_accumulation_steps= int(cfg.get("grad_acc_steps", 1))
grad_clip                  = float(cfg.get("grad_clip", 1.0))
decay_lr                   = True
init_from                  = 'scratch'

# System settings
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype      = 'bfloat16' if (device.startswith('cuda') and torch.cuda.is_bf16_supported()) else 'float16'
compile_flag = True
out_dir    = 'out'
# -----------------------------------------------------------------------------

def get_batch(split):
    data_dir = os.path.join('data', 'shakespeare')
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device.startswith('cuda'):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, ctx):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # W&B init
    wandb.init(
        project="mini-gpt2-pretraining",
        config=cfg,
        save_code=True,
    )

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Model init
    model_args = dict(
        n_layer    = n_layer,
        n_head     = n_head,
        n_embd     = n_embd,
        block_size = block_size,
        bias       = True,
        vocab_size = vocab_size,
        dropout    = dropout,
    )
    meta_path = os.path.join('data', 'shakespeare', 'meta.pkl')
    if os.path.exists(meta_path):
        meta = pickle.load(open(meta_path, 'rb'))
        model_args['vocab_size'] = meta.get('vocab_size', vocab_size)

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)

    if compile_flag and hasattr(torch, 'compile'):
        print("Compiling model…")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = learning_rate,
        betas        = (0.9, 0.99),
        weight_decay = weight_decay,
        foreach      = True,
    )
    scaler = GradScaler(enabled=(dtype == 'float16'))

    iter_num      = 0
    best_val_loss = float('inf')
    t0 = time.time()
    print(f"Training on {device} | tokens/iter={batch_size * block_size:,}")

    pbar = trange(iter_num, max_iters, desc="Training", dynamic_ncols=True)
    for iter_num in pbar:
        # LR update
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Gradient accumulation
        X, Y = get_batch('train')
        for _ in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging every log_interval
        if iter_num % log_interval == 0:
            lossf = loss.detach().item() * gradient_accumulation_steps
            dt = (time.time() - t0) * 1000
            pbar.set_postfix({
                "loss": f"{lossf:.4f}",
                "lr": f"{lr:.2e}",
                "dt": f"{dt:.0f}ms"
            })
            wandb.log({
                "iter": iter_num,
                "train/loss_step": lossf,
                "lr": lr,
            })
            t0 = time.time()

        # Eval & W&B logging
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, ctx)
            train_loss = losses['train']
            val_loss   = losses['val']
            train_ppl  = math.exp(train_loss)
            val_ppl    = math.exp(val_loss)

            print(f"\nstep {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            wandb.log({
                "iter":        iter_num,
                "train/loss":  train_loss,
                "val/loss":    val_loss,
                "train/ppl":   train_ppl,
                "val/ppl":     val_ppl,
                "lr":          lr,
            })

            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                ckpt = {
                    'model_args':   model_args,
                    'model':        model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'iter_num':     iter_num,
                    'best_val_loss': best_val_loss,
                }
                ckpt_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint → {ckpt_path}")

    print("🎉 Training complete.")
    # === Save final model weights-only checkpoint for inference ===
    weights_path = os.path.join(out_dir, f"model_weights_{iter_num}.pt")
    final_weights = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(final_weights, weights_path)
    print(f"✅ Saved weights-only checkpoint → {weights_path}")


if __name__ == "__main__":

    if "--resave_weights" in sys.argv:
        # Automatically get latest checkpoint if you want
        ckpt_files = sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")))
        if not ckpt_files:
            raise FileNotFoundError("❌ No checkpoint files found in 'out/'")
        path = ckpt_files[-1]  # or replace with fixed path like "ckpt_1750.pt"

        print(f"🔁 Loading checkpoint → {path}")
        ckpt = torch.load(path, map_location=device)

        # Rebuild model
        model_args = ckpt["model_args"]
        config = GPTConfig(**model_args)
        model = GPT(config).to(device)

        # Safely fix keys from compiled model
        raw_sd = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in raw_sd.keys()):
            raw_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        model.load_state_dict(raw_sd)

        # Save clean weights
        weights_path = os.path.join(out_dir, "model_weights.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"✅ Resaved model weights-only to {weights_path}")
        sys.exit()

    main()

