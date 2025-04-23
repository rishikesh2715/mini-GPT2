# scripts/train.py

import os
import time
import math
import glob
import yaml
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from model import GPT, GPTConfig
from utils.plots import LiveMetricsPlot



# ======  Hyperparam & Config Loader ======
def load_config(path="config/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ======  Custom Dataset for .bin files ======
class TokenDataset(Dataset):
    def __init__(self, path, block_size):
        data = np.fromfile(path, dtype=np.uint16)
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ======  Train One Epoch ======
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"🚀 Epoch {epoch+1}", leave=False)

    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # update tqdm description every step
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)




# ====== Eval Mode on Val Set ======
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ====== Get Latest Checkpoint ======
def get_latest_checkpoint():
    ckpt_files = glob.glob("checkpoints/gpt_epoch*.pt")
    if not ckpt_files:
        return None, 0
    ckpt_files.sort(key=os.path.getmtime)
    latest = ckpt_files[-1]
    epoch = int(os.path.splitext(os.path.basename(latest))[0].split("epoch")[-1])
    return latest, epoch

# ====== Training Main ======
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============ Load Data ============
    train_dataset = TokenDataset("data/shakespeare/train.bin", cfg["block_size"])
    val_dataset = TokenDataset("data/shakespeare/val.bin", cfg["block_size"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    # ============ Init Model ============
    gpt_cfg = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"]
    )
    model = GPT(gpt_cfg).to(device)

    # ============ Optimizer ============
    cfg["learning_rate"] = float(cfg["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    best_val_loss = float("inf")  # Initialize to infinity for best checkpoint saving

    # ========= Resume from Checkpoint =========
    start_epoch = 0
    latest_ckpt, saved_epoch = get_latest_checkpoint()
    if latest_ckpt:
        print(f"⚡ Resuming from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt))
        start_epoch = saved_epoch

    plotter = LiveMetricsPlot()
    # ============ Train Loop ============
    for epoch in range(start_epoch, cfg["epochs"]):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        end = time.time()

        train_acc, val_acc = 0,0
        plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)

        print("="*60)
        print(f"✅ Epoch {epoch+1}/{cfg['epochs']} completed!")
        print(f"📉 Avg Train Loss: {train_loss:.4f}")
        print(f"🧪 Val Loss     : {val_loss:.4f}")
        print(f"⏱️ Time taken   : {end-start:.2f} seconds")
        print("="*60)


        # Save best checkpoint only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_gpt_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved better model at {ckpt_path} (val_loss: {val_loss:.4f})")


if __name__ == "__main__":
    main()
