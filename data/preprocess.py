# preprocess.py
import os
import requests
from tokenizer import GPT2Tokenizer
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data/shakespeare"
RAW_FILE = os.path.join(DATA_DIR, "input.txt")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
TRAIN_FRACTION = 0.9


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_FILE):
        print("Downloading Tiny Shakespeare...")
        with open(RAW_FILE, 'w', encoding='utf-8') as f:
            f.write(requests.get(DATA_URL).text)
    else:
        print("Dataset already downloaded.")

def prepare_data():
    download_dataset()

    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        data = f.read()

    tokenizer = GPT2Tokenizer()
    tokens = tokenizer.encode(data)

    n = int(len(tokens) * TRAIN_FRACTION)
    train_ids = tokens[:n]
    val_ids = tokens[n:]

    np.array(train_ids, dtype=np.uint16).tofile(TRAIN_BIN)
    np.array(val_ids, dtype=np.uint16).tofile(VAL_BIN)

    print(f"Training tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(val_ids)}")

if __name__ == "__main__":
    prepare_data()