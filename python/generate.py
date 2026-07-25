"""Generate text from a trained checkpoint.

    python3 generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
    python3 generate.py --interactive       # type prompts in a loop
"""

import argparse
import sys

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"])


def stream(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    print(prompt, end="", flush=True)
    for token in model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        print(tokenizer.decode([token]), end="", flush=True)
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    if args.interactive:
        print("type a prompt and press enter (ctrl-d to quit)")
        for line in sys.stdin:
            stream(model, tokenizer, line.rstrip("\n"), args.max_new_tokens,
                   args.temperature, args.top_k, device)
    else:
        stream(model, tokenizer, args.prompt, args.max_new_tokens,
               args.temperature, args.top_k, device)


if __name__ == "__main__":
    main()
