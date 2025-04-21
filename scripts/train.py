import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
import os
import time
import json
from models.gpt2 import MiniGPT2 #INCOMPLETE
from utils.dataset import TextDataset, collate_fn
from utils.logger import Logger
from config.train_config import get_config


def train():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = TextDataset(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = MiniGPT2(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["num_epochs"] * len(dataloader)
    )

    logger = Logger(os.path.join(config["results_path"], "training_log.txt"))

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        logger.log(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

        # Save checkpoint
        checkpoint_path = os.path.join(config["checkpoint_path"], f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete!")


"""

# ✅ TODO:
│   1. Load config and preprocessed data
│   2. Initialize model, loss (CrossEntropy), optimizer (Adam)
│   3. Create data loader with batching logic
│   4. Implement training loop:
│      - forward -> loss -> backward -> step
│      - log and save metrics every N steps
│   5. Save checkpoint periodically (model + optimizer)
│   6. Support resume from checkpoint


"""

