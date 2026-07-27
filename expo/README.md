# PocketGPT in Expo Go

Runs the PocketGPT web app inside an Expo Go container: an app icon, a
fullscreen UI with no browser chrome, and a free install with no Apple
Developer account.

## Read this first

**Expo Go cannot run larger models than Safari can.** Expo Go executes
JavaScript only — it cannot load llama.cpp's native code — so the app here is a
WebView. That is the same WebKit engine Safari uses, with the same memory
budget of roughly 1–2 GB, which means models up to about 1.7B parameters.

If the goal is 3B or larger, this is the wrong tool: use the native app in
[`docs/TESTFLIGHT.md`](../docs/TESTFLIGHT.md), which gets a native memory
budget plus the increased-memory entitlement.

| | Safari (PWA) | Expo Go (this) | Native app (TestFlight) |
|---|---|---|---|
| Cost | free | free | $99/year |
| Needs a computer | no | no (published from CI) | no |
| Largest practical model | ~1.7B | ~1.7B | 3B+ |
| App icon, fullscreen | yes (Add to Home Screen) | yes | yes |

## Option A — no computer: publish from CI, open in Expo Go

1. Create a free account at [expo.dev](https://expo.dev) (works in Safari).
2. Create an access token: **Account settings → Access tokens → Create token**.
3. Add it to this repository as the secret `EXPO_TOKEN`
   (**Settings → Secrets and variables → Actions**).
4. Run **Actions → Publish to Expo Go → Run workflow**.
5. Install **Expo Go** from the App Store, sign in with the same account, and
   the project appears under **Projects**. Open it.

The workflow prints the project URL at the end; opening that link on the phone
launches it in Expo Go directly.

## Option B — with a computer

```bash
npx create-expo-app@latest pocketgpt --template blank
cd pocketgpt
npx expo install react-native-webview react-native-safe-area-context expo-status-bar
cp /path/to/llm-ios/expo/App.js .
cp /path/to/llm-ios/expo/app.json .
npx expo start
```

Scan the QR code with Expo Go.

## Pointing it at a different build

`App.js` loads `APP_URL`, which defaults to the deployed site. Change that
constant to test a branch or a locally served copy.
