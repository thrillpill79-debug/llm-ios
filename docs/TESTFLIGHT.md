# Getting the native app onto your iPhone via TestFlight

This is the only route that needs **no computer at all** — every step below is
done in Safari on the phone, and afterwards new builds arrive over the air in
the TestFlight app.

Why bother, when the web app exists: a Safari tab gets a much smaller memory
budget than a native app. PocketGPT in the browser is comfortable up to about
1–1.5B parameters; the native app, with the increased-memory entitlement,
handles 3B and larger.

Cost: **$99/year** for the Apple Developer Program. There is no free TestFlight.

---

## 1. Enrol in the Apple Developer Program

Install the **Apple Developer** app from the App Store, sign in with your Apple
ID, and complete enrolment there (Account → Enrol). It needs your legal name
and a payment method. Approval is usually minutes to a couple of days.

You will need two-factor authentication on the Apple ID — TestFlight and the
API both require it.

## 2. Create an App Store Connect API key

In Safari, go to **appstoreconnect.apple.com** → **Users and Access** →
**Integrations** → **App Store Connect API** → **Team Keys**.

1. Tap **+**, name it e.g. `github-actions`, give it the **App Manager** role.
2. Download the `AuthKey_XXXXXXXXXX.p8` file. **You can only download it once** —
   save it somewhere you can copy text from (Files, Notes, iCloud Drive).
3. Note the **Key ID** (the `XXXXXXXXXX` part) and the **Issuer ID** shown
   above the key list.

Then find your **Team ID**: **developer.apple.com/account** → Membership
details → Team ID (10 characters).

## 3. Register the app record

In App Store Connect → **Apps** → **+** → **New App**:

- Platform: iOS
- Name: anything not already taken (e.g. `LlamaChat by <yourname>`)
- Bundle ID: create a new one, e.g. `dev.llmios.llamachat` — but bundle IDs are
  globally unique, so if it is taken use something like
  `com.<yourname>.llamachat` and pass that when you run the workflow
- SKU: any string, e.g. `llamachat-1`

## 4. Add four secrets to the repository

In Safari: **github.com/thrillpill79-debug/llm-ios** → **Settings** →
**Secrets and variables** → **Actions** → **New repository secret**. Add:

| Secret name | Value |
|---|---|
| `APPSTORE_ISSUER_ID` | the Issuer ID from step 2 |
| `APPSTORE_KEY_ID` | the Key ID from step 2 |
| `APPSTORE_PRIVATE_KEY` | the **entire contents** of the `.p8` file, including the `-----BEGIN PRIVATE KEY-----` and `-----END PRIVATE KEY-----` lines |
| `APPLE_TEAM_ID` | the 10-character Team ID |

## 5. Run the workflow

**Actions** tab → **TestFlight** → **Run workflow**:

- **app**: `LlamaChat` (or `TinyLLM` for the model you trained)
- **bundle_id**: leave blank to use `dev.llmios.llamachat`, or type the one you
  registered in step 3

The first run takes 30–60 minutes because it compiles llama.cpp for iOS from
source; that result is cached, so later runs take a few minutes. Xcode
registers the bundle ID, enables the entitlements and creates the provisioning
profile automatically using your API key — there is nothing to click.

## 6. Install on your iPhone

Install **TestFlight** from the App Store and sign in with the same Apple ID.
Apple processes each upload for roughly 5–15 minutes, then the build appears
under your app. Tap **Install**. Builds stay valid for 90 days, and every later
workflow run shows up as an update.

---

## If something fails

- **"Missing repository secrets"** — step 4 is incomplete; the workflow checks
  before doing any work.
- **Bundle ID already exists / not owned by your team** — pick a unique one and
  pass it in the `bundle_id` input.
- **Provisioning fails mentioning `increased-memory-limit`** — that entitlement
  lives in `ios/LlamaChat/LlamaChat.entitlements`. Removing that key lets the
  build through; the app then runs with the standard memory cap, so stay with
  1.5B-class models.
- **Export-compliance question in App Store Connect** — the app declares
  `ITSAppUsesNonExemptEncryption = false`, so this should not appear. If it
  does, answer that the app uses no non-exempt encryption.
