"""Verify an export by re-running the model in pure NumPy from the exported files.

    python3 verify_export.py --checkpoint checkpoints/best.pt --export ../model/export

This NumPy forward pass reads config.json / manifest.json / weights.bin exactly
the way the Swift engine does, and is the line-by-line reference for
ios/TinyLLM/Engine/TinyGPT.swift. It compares logits against the PyTorch model
on random sequences and fails loudly if they diverge.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from model import GPT, GPTConfig


class NumpyGPT:
    def __init__(self, export_dir: str):
        with open(os.path.join(export_dir, "config.json")) as f:
            self.cfg = json.load(f)
        with open(os.path.join(export_dir, "manifest.json")) as f:
            manifest = json.load(f)
        raw = np.fromfile(os.path.join(export_dir, "weights.bin"), dtype="<f4")
        self.w = {
            e["name"]: raw[e["offset"]:e["offset"] + e["count"]].reshape(e["shape"])
            for e in manifest
        }

    @staticmethod
    def layernorm(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    @staticmethod
    def softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def forward(self, ids):
        """ids: list[int], length <= block_size. Returns logits [T, vocab]."""
        cfg, w = self.cfg, self.w
        T = len(ids)
        n_head = cfg["n_head"]
        hd = cfg["n_embd"] // n_head

        x = w["tok_emb.weight"][ids] + w["pos_emb.weight"][:T]
        for i in range(cfg["n_layer"]):
            p = f"blocks.{i}."
            h = self.layernorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, n_head, hd).transpose(1, 0, 2)
            k = k.reshape(T, n_head, hd).transpose(1, 0, 2)
            v = v.reshape(T, n_head, hd).transpose(1, 0, 2)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            att[:, mask] = -np.inf
            att = self.softmax(att)
            y = (att @ v).transpose(1, 0, 2).reshape(T, cfg["n_embd"])
            x = x + y @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"]

            h = self.layernorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = self._gelu(h @ w[p + "mlp.fc.weight"].T + w[p + "mlp.fc.bias"])
            x = x + h @ w[p + "mlp.proj.weight"].T + w[p + "mlp.proj.bias"]

        x = self.layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
        return x @ w["tok_emb.weight"].T

    @staticmethod
    def _gelu(x):
        # exact GELU (erf form), matching torch.nn.functional.gelu's default;
        # numpy has no erf, so vectorize math.erf
        import math
        erf = np.vectorize(math.erf)
        return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--export", default=os.path.join(os.path.dirname(__file__), "..", "model", "export"))
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = GPTConfig(**ckpt["config"])
    torch_model = GPT(config)
    torch_model.load_state_dict(ckpt["model"])
    torch_model.eval()

    np_model = NumpyGPT(args.export)

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(args.trials):
        T = int(rng.integers(4, config.block_size + 1))
        ids = rng.integers(0, config.vocab_size, size=T).tolist()
        with torch.no_grad():
            ref, _ = torch_model(torch.tensor([ids], dtype=torch.long))
        ref = ref[0].numpy()
        got = np_model.forward(ids)
        diff = float(np.abs(ref - got).max())
        worst = max(worst, diff)
        print(f"trial {trial}: T={T:3d}  max|logit diff| = {diff:.2e}")

    if worst > 1e-3:
        print(f"FAIL: exported weights diverge from checkpoint (max diff {worst:.2e})")
        sys.exit(1)
    print(f"OK: NumPy forward from exported weights matches PyTorch (max diff {worst:.2e})")


if __name__ == "__main__":
    main()
