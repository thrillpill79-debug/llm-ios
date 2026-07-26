# Building and installing the apps with no Mac

You don't need a Mac for any of this. GitHub's free macOS runners compile the
apps; your iPhone runs them; any Windows or Linux PC handles the one
installation step — and if you have **only an iPhone**, see the next section
first.

## Only an iPhone? Two paths

**Path 1 — the web app (free, works right now).** Every push deploys to
GitHub Pages:

> **https://thrillpill79-debug.github.io/llm-ios/**

Open it in Safari on your iPhone → Share → **Add to Home Screen**. No signing,
no computer, no money, no expiry. Two things live there:

- **PocketGPT** — a genuine instruct-tuned assistant running through
  llama.cpp compiled to WebAssembly. Pick a model (0.5B recommended for a
  phone browser tab), it downloads once, and from then on it is offline and
  private. Because a browser tab gets a much smaller memory budget than a
  native app, stick to 0.5B-class models on iPhone; the native LlamaChat app
  below is the way to run 1–3B.
- **Shakespeare** — the from-scratch model trained in this repo, verified
  against the PyTorch reference by CI on every deploy.

**Path 2 — the native apps via TestFlight ($99/year, no computer ever).**
This is the route to run 3B-class models, which a browser tab cannot fit.
The pipeline is already built: enrol from your iPhone, add four secrets, and
run the **TestFlight** workflow. Step-by-step instructions, all doable in
Safari, are in [TESTFLIGHT.md](TESTFLIGHT.md).

A middle option: UDID signing services (Signulous, AppDB and similar,
~$20/year) sign the CI-built IPAs entirely from Safari on the phone — you
upload the IPA, they sign it against their developer account with your
device's UDID, and you install over the air. Third-party, so read their
terms, but no computer is involved.

## 1. Let CI build the apps

**TinyLLM** builds automatically on every push: repo → **Actions** tab →
latest **CI** run → download the `TinyLLM-unsigned-ipa` artifact.
The same run also compiles the engine with the official Swift toolchain on
Linux and checks its output numerically against the trained model, so a green
run means the Swift genuinely works, not just that it compiles.

**LlamaChat** is a manual trigger (its first run spends ~30–60 minutes
compiling llama.xcframework from source, cached afterwards): **Actions** tab →
**LlamaChat IPA** → **Run workflow** → wait → download
`LlamaChat-unsigned-ipa`.

Each artifact zip contains an unsigned `.ipa` — a complete app that just needs
a signature your iPhone will accept.

> **Minutes budget:** if this repo is private, macOS runner time counts 10×
> against the free 2,000 min/month (≈200 macOS minutes). The LlamaChat
> first build fits, but barely — making the repo public removes all limits.

## 2. Sideload the IPA onto your iPhone

Apple requires apps to be signed. Without a Mac, sign with your Apple ID from
a PC:

**Windows:** [Sideloadly](https://sideloadly.io) (simplest) or
[AltStore](https://altstore.io) with AltServer. Plug in your iPhone, pick the
`.ipa`, sign in with a free Apple ID, install.

**Linux:** [Sideloader](https://github.com/Dadoum/Sideloader) is the
open-source equivalent.

Free-Apple-ID limitations (Apple's rules, not the tools'):
- the app expires after **7 days** — re-sideload to refresh it
- at most 3 sideloaded apps at a time
- the first launch needs **Settings → General → VPN & Device Management** →
  trust your developer certificate
- the *Increased Memory Limit* entitlement is ignored, so in LlamaChat stick
  to the 0.5B/1.5B models (the 3B needs that entitlement on a base iPhone 15)

## 3. The no-expiry upgrade path (optional, $99/year)

An [Apple Developer Program](https://developer.apple.com/programs/) membership
(enrollable from your iPhone, no Mac) unlocks:
- **TestFlight distribution built in CI** — GitHub Actions signs and uploads
  the app; you install and update over the air from the TestFlight app, no PC
  in the loop at all, and builds last 90 days
- the Increased Memory Limit entitlement (3B models on a base iPhone 15)

If you enroll, say so — wiring App Store Connect keys into the workflow is a
small follow-up.

## What CI verifies on every push

| Check | Where | What it proves |
|---|---|---|
| `swift test` | Linux, official toolchain | engine compiles; logits match the NumPy/PyTorch reference to <5e-3; greedy generation reproduces the reference token-for-token; UTF-8 streaming decoder correct |
| TinyLLM build | macOS runner | the full SwiftUI app compiles for iOS device |
| LlamaChat build | macOS runner (manual) | llama.cpp compiles for iOS + the chat app compiles against it |
