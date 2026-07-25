"""Train your LLM on any plain-text file.

    python3 train.py                       # trains on Tiny Shakespeare (auto-downloaded)
    python3 train.py --data my_corpus.txt  # trains on your own text
    python3 train.py --steps 3000 --n-layer 6 --n-embd 192   # bigger model

Checkpoints go to checkpoints/ (best-so-far by validation loss, plus final).
"""

import argparse
import math
import os
import time
import urllib.request

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_corpus(path: str) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading Tiny Shakespeare -> {path}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=40):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, max_steps, base_lr, warmup=100):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * base_lr + 0.45 * base_lr * (1 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="path to a UTF-8 text file (default: Tiny Shakespeare)")
    p.add_argument("--out", default="checkpoints", help="output directory")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    text = load_corpus(args.data)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = max(1000, int(0.1 * len(ids)))
    splits = {"train": ids[:-n_val], "val": ids[-n_val:]}
    print(f"corpus: {len(text):,} chars, vocab: {tokenizer.vocab_size}, "
          f"train/val tokens: {len(splits['train']):,}/{len(splits['val']):,}")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"model: {model.num_params():,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def save(path, val_loss):
        torch.save({
            "model": model.state_dict(),
            "config": config.__dict__,
            "chars": tokenizer.chars,
            "val_loss": val_loss,
        }, path)

    best_val = float("inf")
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(splits["train"], config.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            losses = estimate_loss(model, splits, config.block_size, args.batch_size, device)
            print(f"eval  {step + 1:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}", flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(os.path.join(args.out, "best.pt"), best_val)
                print(f"      saved {args.out}/best.pt (val {best_val:.4f})", flush=True)

    save(os.path.join(args.out, "final.pt"), best_val)
    print(f"done in {time.time() - t0:.0f}s. best val loss {best_val:.4f}. "
          f"try: python3 generate.py --checkpoint {args.out}/best.pt")


if __name__ == "__main__":
    main()
