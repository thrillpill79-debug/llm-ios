# llm-ios — your own LLM, trained by you, running on your iPhone

A complete, dependency-light pipeline for owning a language model end to end:

1. **`python/`** — a GPT-style transformer written from scratch in ~150 lines of
   PyTorch. Train it on any text file you like.
2. **`model/export/`** — your trained weights in a simple flat binary format
   (a ready-to-use demo model trained on Shakespeare is checked in).
3. **`ios/`** — a SwiftUI app with a pure-Swift inference engine (no
   frameworks, no server, no API keys) that runs the model entirely on-device.

The whole model is small enough to read in one sitting: token + positional
embeddings, a few pre-norm transformer blocks (causal self-attention + GELU
MLP), a final LayerNorm, and a weight-tied LM head.

## Quick start (Python)

```bash
cd python
pip install torch numpy

# train on Tiny Shakespeare (auto-downloads, ~10–20 min on a laptop CPU)
python3 train.py

# talk to your model
python3 generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
python3 generate.py --interactive
```

Train on **your own text** instead — song lyrics, your notes, a book, code:

```bash
python3 train.py --data path/to/your_corpus.txt
```

Anything over ~1 MB of text gives noticeably better results. Scale the model
up with `--n-layer 6 --n-embd 192 --steps 5000` if you have a GPU
(`--device cuda`) or Apple Silicon (`--device mps`).

## Put it on your iPhone

```bash
cd python
python3 export_ios.py --checkpoint checkpoints/best.pt   # writes ../model/export/
python3 verify_export.py --checkpoint checkpoints/best.pt  # sanity-checks the export
```

Then follow [`ios/README.md`](ios/README.md) — in short: create an iOS App
project in Xcode, drag in `ios/TinyLLM/` sources and the four files from
`model/export/`, and run. Generation streams token-by-token, fully offline.

A demo model (~0.8 M parameters, char-level, trained on Shakespeare) is
already committed in `model/export/`, so the iOS app works before you train
anything.

## How it fits together

```
your_corpus.txt ──▶ train.py ──▶ checkpoints/best.pt
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
              generate.py (PyTorch)          export_ios.py
              chat on your computer                  │
                                                     ▼
                                          model/export/{config,vocab,
                                            manifest}.json + weights.bin
                                                     │
                                 verify_export.py    │   (NumPy re-implementation
                                 checks the export ◀─┤    of the forward pass)
                                                     ▼
                                        ios/TinyLLM (Swift engine + SwiftUI)
                                        runs it on your iPhone, offline
```

Three implementations of the same forward pass, kept deliberately in sync:

| | file | role |
|---|---|---|
| PyTorch | `python/model.py` | training + reference |
| NumPy | `python/verify_export.py` | validates the exported binary format |
| Swift | `ios/TinyLLM/Engine/TinyGPT.swift` | on-device inference (KV cache) |

## Export format

`weights.bin` is every tensor as raw little-endian float32, concatenated.
`manifest.json` records each tensor's name, shape, and offset. Linear weights
keep PyTorch's `[out_features, in_features]` row-major layout, so inference is
plain row-dot-products. The LM head shares the token-embedding matrix
(weight tying), so it isn't stored twice.

## Ideas for where to take it

- **BPE tokenizer** (currently char-level) for better sample quality per FLOP
- **Instruction tuning**: fine-tune on `question\nanswer` pairs to make it chatty
- **Quantize to int8** to shrink `weights.bin` 4× for bigger models
- **Metal / MPSGraph** in the Swift engine if you scale past a few million params
