# TinyLLM iOS app

A SwiftUI app that runs your trained model fully on-device. The inference
engine is plain Swift (Accelerate-accelerated where available) — no ML
frameworks, no network access, no API keys.

## Build steps

1. In Xcode: **File → New → Project → iOS → App**.
   - Product name: `TinyLLM`
   - Interface: SwiftUI, Language: Swift
   - Minimum deployment target: iOS 17.0
2. Delete the generated `ContentView.swift` and the generated `<name>App.swift`.
3. Drag the contents of `ios/TinyLLM/` (both `.swift` files and the `Engine/`
   folder) into the project navigator. Check **"Copy items if needed"** and add
   them to the app target.
4. Drag the four files from `model/export/` — `config.json`, `vocab.json`,
   `manifest.json`, `weights.bin` — into the project. Make sure each has
   **Target Membership** checked so they land in the app bundle.
5. Run on the simulator or a device.

Type a prompt (try `ROMEO:` with the bundled Shakespeare model), hit
**Generate**, and watch it stream. The temperature slider trades coherence
for creativity.

## Swapping in your own model

Re-train and re-export on the Python side, then replace the four files from
`model/export/` in the Xcode project. Nothing else changes — the Swift engine
reads all dimensions from `config.json` and the tensor layout from
`manifest.json`.

## Engine notes

- `Engine/TinyGPT.swift` processes one token per forward pass with a per-layer
  KV cache, so generation cost stays flat as output grows.
- The model uses learned absolute positional embeddings, so when the context
  window fills, the engine re-prefills from the most recent half of the window
  rather than sliding the cache.
- `Engine/MathOps.swift` uses vDSP (Accelerate) on Apple platforms and falls
  back to portable loops elsewhere, so the engine also compiles on Linux for
  testing.
- The math mirrors `python/verify_export.py` line by line — if you change the
  architecture in `model.py`, update both.
- If your project uses Swift 6 strict concurrency with default `@MainActor`
  isolation and you hit actor-isolation errors in the engine, set the target's
  **Default Actor Isolation** to `nonisolated` (or language mode to Swift 5);
  the engine types are annotated `Sendable` and run on a background task.
