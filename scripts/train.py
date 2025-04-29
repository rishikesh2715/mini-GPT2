# scripts/train.py
# ------------------------------------------------------------
# Mini-GPT-2 training script with live plotting & performance
# tweaks (AMP, fused AdamW, non-blocking H2D copies, torch.compile)
# ------------------------------------------------------------
import os
import time
import math
import glob
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse

from model         import GPT, GPTConfig
from utils.plots   import LiveMetricsPlot

# ---- Perf flags ------------------------------------------------------------
torch.backends.cudnn.benchmark = True        # fastest CUDNN kernels for static shapes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================ #
#                               Utility helpers                                #
# ============================================================================ #
def load_config(path: str = "config/train_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TokenDataset(Dataset):
    """Memory-mapped dataset of uint16-encoded tokens."""
    def __init__(self, path: str, block_size: int):
        data = np.fromfile(path, dtype=np.uint16)
        self.data        = torch.tensor(data, dtype=torch.long)
        self.block_size  = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x     = chunk[:-1]
        y     = chunk[1:]
        return x, y


# ============================================================================ #
#                             Training / Evaluation                            #
# ============================================================================ #
def train_one_epoch(model, dataloader, optimizer, scaler, epoch,
                    plotter=None, log_every: int = 200):
    """Train for a single epoch with mixed precision + live plotting."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(dataloader, desc=f"🚀 Epoch {epoch + 1}", leave=False)

    for step, (x, y) in enumerate(pbar, 1):
        # non-blocking host→device copies (needs pin_memory=True in loader)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forward pass in FP16
        with torch.autocast("cuda", dtype=torch.float16):
            logits, loss = model(x, y)

        # backward + optimizer step (scaled)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # running stats
        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == y).sum().item()
        total      += y.numel()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # mini-batch live plot
        if plotter and step % log_every == 0:
            frac_epoch   = epoch + step / len(dataloader)
            running_loss = total_loss / step
            plotter.update(frac_epoch, running_loss, running_loss)

    avg_loss  = total_loss / len(dataloader)
    train_acc = correct / total
    return avg_loss, train_acc


@torch.no_grad()
def evaluate(model, dataloader):
    """Validation loop with mixed precision."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in dataloader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):
            logits, loss = model(x, y)

        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == y).sum().item()
        total      += y.numel()

    return total_loss / len(dataloader), correct / total


# cosine-decay LR with warm-up
def lr_schedule(epoch, *, warmup=5, base_lr=3e-4, min_lr=1e-5, total_epochs=2):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = epoch - warmup
    T = total_epochs - warmup
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / T))


def latest_checkpoint():
    ckpts = glob.glob("checkpoints/gpt_epoch*.pt")
    if not ckpts:
        return None, 0
    ckpts.sort(key=os.path.getmtime)
    path  = ckpts[-1]
    epoch = int(os.path.splitext(os.path.basename(path))[0].split("epoch")[-1])
    return path, epoch


# ============================================================================ #
#                                      Main                                    #
# ============================================================================ #
def main(no_plot: bool = False):
    cfg = load_config()
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["weight_decay"]  = float(cfg.get("weight_decay", 0.01))
    cfg["min_lr"]        = float(cfg.get("min_lr", 1e-5))

    # ------------------- Data -------------------
    train_ds = TokenDataset("data/shakespeare/train.bin", cfg["block_size"])
    val_ds   = TokenDataset("data/shakespeare/val.bin",   cfg["block_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = 8,          # tune for CPU
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"],
        num_workers = 4,
        pin_memory  = True,
    )

    # ------------------- Model -------------------
    gpt_cfg = GPTConfig(
        block_size = cfg["block_size"],
        vocab_size = cfg["vocab_size"],
        n_layer    = cfg["n_layer"],
        n_head     = cfg["n_head"],
        n_embd     = cfg["n_embd"],
        dropout    = cfg["dropout"],
    )
    model = GPT(gpt_cfg).to(DEVICE)

    # optional Graph-capture speed-up (PyTorch 2+)
    if (
        torch.__version__ >= "2.0.0"
        and DEVICE.type == "cuda"
        and os.name != "nt"
    ):
        model = torch.compile(model)

    # ------------------- Optimizer / AMP -----------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"],
        betas        = (0.9, 0.95),
        foreach      = True,          # fused CUDA kernel in PyTorch 2.x
    )
    scaler = torch.amp.GradScaler()

    # ------------------- Resume (optional) ---------
    best_val_loss = float("inf")
    start_epoch   = 0
    ckpt_path, start_epoch = latest_checkpoint()
    if ckpt_path:
        print(f"⚡ Resuming from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # ------------------- Live plot -----------------
    plotter = None if no_plot else LiveMetricsPlot()

    # ------------------- Training loop -------------
    print(f"Device: {DEVICE} | CUDA: {torch.version.cuda} | Torch: {torch.__version__}")
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for epoch in range(start_epoch, cfg["epochs"]):
        # adjust LR
        lr = lr_schedule(
            epoch,
            warmup        = cfg.get("warmup_epochs", 5),
            base_lr       = cfg["learning_rate"],
            min_lr        = cfg["min_lr"],
            total_epochs  = cfg["epochs"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        print(f"📉 LR → {lr:.2e}")

        # pick ~50 points per epoch for plotting
        log_every = max(1, len(train_loader) // 50)

        # ---- train + val -----------------------------------------
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler,
            epoch,
            plotter   = plotter,
            log_every = 10,
        )
        val_loss, val_acc = evaluate(model, val_loader)
        elapsed = time.time() - t0

        # epoch-level plot update
        if plotter:
            plotter.update(epoch + 1, tr_loss, val_loss, tr_acc, val_acc)

        # ---- console log ----------------------------------------
        tr_ppl  = math.exp(tr_loss)
        val_ppl = math.exp(val_loss)
        print("=" * 65)
        print(f"✓ Epoch {epoch+1}/{cfg['epochs']}  "
              f"Train loss {tr_loss:.4f}  |  Val loss {val_loss:.4f}")
        print(f"  Train ppl {tr_ppl:8.2f} | Val ppl {val_ppl:8.2f}")
        print(f"  Acc: train {tr_acc*100:5.2f}% | val {val_acc*100:5.2f}%")
        print(f"  ⏱  {elapsed:.1f}s")
        print("=" * 65)

        # save best ckpt
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/best_gpt_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model → {save_path}")

    # end for epoch
    print("🎉 Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Disable live Matplotlib window (still logs CSV)."
    )
    args = parser.parse_args()
    main(no_plot=args.no_plot)
