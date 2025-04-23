# scripts/generate.py

import torch
import argparse
from model import GPT, GPTConfig
from data.tokenizer import GPT2Tokenizer
import os
import glob
import yaml

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, top_k=None, device='cpu'):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]  # crop to block size
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # take last step
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0].tolist())

def load_latest_checkpoint():
    ckpt_files = glob.glob("checkpoints/*.pt")
    if not ckpt_files:
        raise FileNotFoundError("❌ No checkpoints found in 'checkpoints/'")
    ckpt_files.sort(key=os.path.getmtime)
    return ckpt_files[-1]


def load_config(path="config/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="To be or not to be", help="Prompt to begin generation")
    parser.add_argument("--tokens", type=int, default=100, help="How many tokens to generate")
    parser.add_argument("--top_k", type=int, default=None, help="Use top-k sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer()

    cfg = load_config()
    config = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"]
    )
 # same config as training
    model = GPT(config)
    model.load_state_dict(torch.load(load_latest_checkpoint(), map_location=device))
    model.to(device)

    print(f"📜 Prompt: {args.prompt}")
    output = generate(model, tokenizer, args.prompt, max_new_tokens=args.tokens, top_k=args.top_k, device=device)
    print("\n🧠 Generated:\n")
    print(output)

if __name__ == "__main__":
    main()
