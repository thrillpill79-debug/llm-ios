# llm-ios — your own LLM, trained by you, running on your iPhone

Two tiers, one repo:

**Tier 1 — TinyLLM: a model you build and train yourself.**
1. **`python/`** — a GPT-style transformer written from scratch in ~150 lines of
   PyTorch. Train it on any text file you like.
2. **`model/export/`** — your trained weights in a simple flat binary format
   (a ready-to-use demo model trained on Shakespeare is checked in).
3. **`ios/TinyLLM/`** — a SwiftUI app with a pure-Swift inference engine (no
   frameworks, no server, no API keys) that runs the model entirely on-device.

**Tier 2 — LlamaChat: assistant-class pretrained models, still fully on-device.**
[`ios/LlamaChat/`](ios/LlamaChat/README.md) is a chat app built on llama.cpp
that runs real instruct models (Qwen2.5 0.5B/1.5B, Llama 3.2 3B, or any GGUF
you point it at) with Metal acceleration — roughly the practical ceiling of an
iPhone 15. Tier 1 is for understanding every line; Tier 2 is for a model you
can actually talk to.

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

**No Mac?** See [`docs/NO-MAC.md`](docs/NO-MAC.md): GitHub Actions builds
both apps on free macOS runners (unsigned IPAs as artifacts, numerically
tested on every push), and a Windows/Linux PC sideloads them onto your
iPhone.

**Only an iPhone?** Everything also runs in Safari, no install required:
**https://thrillpill79-debug.github.io/llm-ios/**

- **PocketGPT** (main page) — a real instruct-tuned assistant (Qwen2.5 0.5B,
  SmolLM2 360M, Llama 3.2 1B, or any GGUF URL) running via llama.cpp compiled
  to WebAssembly. The model downloads once, is stored on the phone, and then
  works offline. Nothing is sent anywhere.
- **Shakespeare** (`shakespeare.html`) — the tiny model *you* trained, chatting
  in verse.

Add to Home Screen and it behaves like an installed app.

**Want the native app instead?** A browser tab is memory-limited to roughly
1–1.5B parameters; the native LlamaChat app runs 3B-class models. With an
Apple Developer membership it installs over the air, no computer involved —
see [`docs/TESTFLIGHT.md`](docs/TESTFLIGHT.md).

**Expo Go wrapper** ([`expo/`](expo/README.md)) — the same web app in an Expo Go
container: app icon, fullscreen, free, no Apple account. It is a WebView, so
the memory ceiling is identical to Safari's; it buys presentation, not
capacity.

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
- **Fine-tune a pretrained model** on your own data and run it in LlamaChat —
  the bridge between the two tiers
