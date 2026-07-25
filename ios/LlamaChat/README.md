# LlamaChat — assistant-class models on your iPhone

The second tier of this repo: instead of the tiny from-scratch model, this app
runs real pretrained instruct models (Qwen2.5, Llama 3.2, or any GGUF) fully
on-device via [llama.cpp](https://github.com/ggml-org/llama.cpp), with Metal
GPU acceleration. A 1.5B model streams ~20 tok/s on an iPhone 15; nothing ever
leaves the phone.

The app downloads a model on first launch (0.4–2 GB, one time) — GGUF files
are far too large to bundle or commit.

## Build steps

### 1. Build llama.xcframework (once, ~5 minutes)

On your Mac:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
./build-xcframework.sh
```

The result lands in `build-apple/llama.xcframework`. (The Swift code here was
written against the llama.cpp API as of the revision vendored in
llama-cpp-python 0.3.34, mid-2026. The core API used — model/context init,
tokenize, decode, sampler chain, chat templates, `llama_memory_*` — is the
stable modern surface, but if a much newer master ever renames something, the
compiler will point at it.)

### 2. Create the Xcode project

1. **File → New → Project → iOS → App**, product name `LlamaChat`,
   Interface: SwiftUI. Minimum deployment target **iOS 17.0** (the
   xcframework itself supports 16.4+).
2. Delete the generated `ContentView.swift` and `<name>App.swift`.
3. Drag the contents of `ios/LlamaChat/` (the `.swift` files and `Engine/`
   folder) into the project. Check "Copy items if needed", add to the target.
4. Drag `llama.xcframework` into the project. In the target's **General →
   Frameworks, Libraries, and Embedded Content**, set it to **Embed & Sign**.
5. In **Signing & Capabilities**, add the **Increased Memory Limit**
   capability. (Not strictly needed for 0.5–1.5B models; required to run 3B
   comfortably on a base iPhone 15.)
6. Run on a device. The simulator works too (CPU-only — Metal is disabled
   there), just slower.

### 3. Pick a model in the app

On first launch the app offers a small curated catalog:

| Model | Size | Notes |
|---|---|---|
| Qwen2.5 0.5B Instruct Q4_K_M | ~0.4 GB | fastest, good pipeline test |
| Qwen2.5 1.5B Instruct Q4_K_M | ~1.0 GB | recommended; ~20 tok/s on iPhone 15 |
| Llama 3.2 3B Instruct Q4_K_M | ~2.0 GB | best quality a base iPhone 15 fits |

There's also a field to paste any direct `.gguf` URL (the catalog URLs point
at Hugging Face and could rot if a repo reorganizes — any GGUF chat model
works, that's the point of the format).

## How the engine works (`Engine/LlamaEngine.swift`)

- Each turn, the full conversation is rendered with the model's own chat
  template (`llama_chat_apply_template`) and tokenized.
- The engine keeps a shadow copy of the tokens in the KV cache, reuses the
  longest common prefix, drops any divergent tail with `llama_memory_seq_rm`,
  and decodes only the new suffix — the same prefix-caching strategy as
  llama.cpp's server, so long chats never re-prefill from scratch.
- Sampling is a min-p → temperature → dist chain (llama.cpp's modern default);
  temperature ≤ 0.05 switches to greedy.
- Token pieces are reassembled through a small UTF-8 streaming decoder so
  emoji/CJK characters split across tokens never render as �.

## Memory guardrails (iPhone 15, 6 GB)

iOS caps an app well below physical RAM (~3–3.5 GB before jetsam). Weights at
Q4_K_M plus a 4096-token KV cache: 0.5B ≈ 0.6 GB, 1.5B ≈ 1.3 GB, 3B ≈ 2.4 GB.
That's why the catalog stops at 3B — a 7–8B model does not fit on a base
iPhone 15 at usable quantization. If you have a 15 Pro (8 GB), 7–8B at Q3/Q4
is possible via the custom-URL field, with the Increased Memory Limit
capability enabled.
