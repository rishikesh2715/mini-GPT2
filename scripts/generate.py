# scripts/generate.py

import torch
import argparse
from model import GPT, GPTConfig
from data.tokenizer import GPT2Tokenizer
import os
import glob
import yaml

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, top_k=None, temperature=1.0, device='cpu'):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]  # crop to block size
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # take last step

        # 🔥 Apply temperature scaling
        logits = logits / temperature

        # 🔽 Apply top-k filtering
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0].tolist())


def load_latest_weights(path="out"):
    weight_files = sorted(glob.glob(os.path.join(path, "model_weights*.pt")))
    if not weight_files:
        raise FileNotFoundError("❌ No weights file found in 'out/' (expected 'model_weights*.pt')")
    return weight_files[-1]


def load_config(path="config/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="To be or not to be", help="Prompt to begin generation")
    parser.add_argument("--tokens", type=int, default=100, help="How many tokens to generate")
    parser.add_argument("--top_k", type=int, default=None, help="Use top-k sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random)")
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

    model = GPT(config)

    # ✅ Load latest weights file directly
    latest_weights = load_latest_weights()
    print(f"🔍 Loading weights → {latest_weights}")
    model.load_state_dict(torch.load(latest_weights, map_location=device))

    model.to(device)
    model.eval()

    print(f"📜 Prompt: {args.prompt}")
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        device=device
    )
    print("\n🧠 Generated:\n")
    print(output)


if __name__ == "__main__":
    main()
