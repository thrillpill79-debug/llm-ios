"""Export a trained checkpoint into the flat format the iOS app loads.

    python3 export_ios.py --checkpoint checkpoints/best.pt --out ../model/export

Produces four files:
    config.json    model hyperparameters
    vocab.json     the character vocabulary (token id = index)
    manifest.json  name, shape, offset (in floats) of every tensor in weights.bin
    weights.bin    all weights as raw little-endian float32, concatenated

Linear weights keep PyTorch's [out_features, in_features] row-major layout,
so an inference engine computes y[o] = sum_i W[o*in + i] * x[i] + b[o].
The LM head is weight-tied to tok_emb and is not exported separately.
"""

import argparse
import json
import os

import numpy as np
import torch

from model import GPT, GPTConfig


def export(checkpoint_path: str, out_dir: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
        }, f, indent=2)

    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump({"chars": ckpt["chars"]}, f)

    # fixed, documented tensor order; the manifest makes lookup order-independent
    names = ["tok_emb.weight", "pos_emb.weight"]
    for i in range(config.n_layer):
        names += [
            f"blocks.{i}.ln1.weight", f"blocks.{i}.ln1.bias",
            f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias",
            f"blocks.{i}.attn.proj.weight", f"blocks.{i}.attn.proj.bias",
            f"blocks.{i}.ln2.weight", f"blocks.{i}.ln2.bias",
            f"blocks.{i}.mlp.fc.weight", f"blocks.{i}.mlp.fc.bias",
            f"blocks.{i}.mlp.proj.weight", f"blocks.{i}.mlp.proj.bias",
        ]
    names += ["ln_f.weight", "ln_f.bias"]

    state = model.state_dict()
    manifest = []
    offset = 0
    chunks = []
    for name in names:
        t = state[name].detach().to(torch.float32).contiguous().numpy()
        manifest.append({"name": name, "shape": list(t.shape), "offset": offset, "count": int(t.size)})
        chunks.append(t.reshape(-1))
        offset += int(t.size)

    weights = np.concatenate(chunks).astype("<f4")
    weights.tofile(os.path.join(out_dir, "weights.bin"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    size_mb = weights.nbytes / 1e6
    print(f"exported {len(manifest)} tensors, {offset:,} floats ({size_mb:.1f} MB) -> {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "model", "export"))
    args = p.parse_args()
    export(args.checkpoint, args.out)
