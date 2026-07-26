// PocketGPT — a real instruct-tuned LLM running fully in the browser via
// llama.cpp compiled to WebAssembly (wllama). Models are downloaded once
// from Hugging Face, stored in the browser's origin-private filesystem, and
// run entirely on-device: no server, no API key, no data leaving the phone.
import { Wllama } from "./vendor/wllama/index.js";

const WASM_PATHS = { default: "./vendor/wllama/wllama.wasm" };

// Filenames are NOT hardcoded: repositories rename and reorganise their quant
// files, and a guessed URL is a dead link. Instead each entry lists candidate
// repos, and the app asks Hugging Face's API which .gguf files actually exist,
// picking the preferred quant from the real listing.
//
// Nothing here is capped or gated — bigger models are simply labelled with what
// they cost, and any GGUF URL can be pasted in.
const CATALOG = [
  {
    name: "Qwen3 1.7B",
    size: "~1.1 GB",
    note: "Best answers per megabyte here. Recommended starting point.",
    repos: [
      "Qwen/Qwen3-1.7B-GGUF",
      "unsloth/Qwen3-1.7B-GGUF",
      "bartowski/Qwen_Qwen3-1.7B-GGUF",
    ],
  },
  {
    name: "Qwen3 0.6B",
    size: "~400 MB",
    note: "Quickest to get running, and stronger than older models twice its size.",
    repos: [
      "Qwen/Qwen3-0.6B-GGUF",
      "unsloth/Qwen3-0.6B-GGUF",
      "bartowski/Qwen_Qwen3-0.6B-GGUF",
    ],
  },
  {
    name: "Gemma 3 1B Instruct",
    size: "~800 MB",
    note: "Google's small model. Clear, well-behaved prose.",
    repos: [
      "unsloth/gemma-3-1b-it-GGUF",
      "ggml-org/gemma-3-1b-it-GGUF",
      "bartowski/google_gemma-3-1b-it-GGUF",
    ],
  },
  {
    name: "Llama 3.2 1B Instruct",
    size: "~810 MB",
    note: "Solid all-rounder, widely tested.",
    repos: [
      "unsloth/Llama-3.2-1B-Instruct-GGUF",
      "bartowski/Llama-3.2-1B-Instruct-GGUF",
      "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
    ],
  },
  {
    name: "Qwen2.5 0.5B Instruct",
    size: "~400 MB",
    note: "Small and dependable. Good fallback if a newer model misbehaves.",
    repos: [
      "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
      "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
      "unsloth/Qwen2.5-0.5B-Instruct-GGUF",
    ],
  },
  {
    name: "Qwen3 4B",
    size: "~2.5 GB",
    note: "Markedly better reasoning. Big download, slower per word.",
    repos: [
      "Qwen/Qwen3-4B-GGUF",
      "unsloth/Qwen3-4B-GGUF",
      "bartowski/Qwen_Qwen3-4B-GGUF",
    ],
  },
  {
    name: "Llama 3.2 3B Instruct",
    size: "~2.0 GB",
    note: "Strong general model at 3B.",
    repos: [
      "unsloth/Llama-3.2-3B-Instruct-GGUF",
      "bartowski/Llama-3.2-3B-Instruct-GGUF",
      "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    ],
  },
  {
    name: "Qwen3 8B",
    size: "~5.0 GB",
    note: "Desktop-class. Unlikely to fit a phone — included because you asked for no limits.",
    repos: [
      "Qwen/Qwen3-8B-GGUF",
      "unsloth/Qwen3-8B-GGUF",
      "bartowski/Qwen_Qwen3-8B-GGUF",
    ],
  },
];

// Preference order for quantisations: best quality-per-byte on a phone first.
// q2_k is last because it degrades answers badly — the thing we least want.
const QUANT_ORDER = [/q4_k_m/i, /q4_k_s/i, /q4_0/i, /q5_k_m/i, /q5_k_s/i,
                     /q3_k_m/i, /q8_0/i, /q6_k/i, /q2_k/i];

const SHARD_RE = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i;

/** List the .gguf files a repo really contains (null if unreachable/gated). */
async function listRepoFiles(repo) {
  try {
    const res = await fetch(
      `https://huggingface.co/api/models/${repo}/tree/main?recursive=true`);
    if (!res.ok) return null;
    const entries = await res.json();
    return entries
      .filter((e) => e.type === "file" && e.path.toLowerCase().endsWith(".gguf"))
      // vision projectors are not standalone models
      .filter((e) => !/mmproj/i.test(e.path))
      .map((e) => ({ path: e.path, size: e.size ?? 0 }));
  } catch {
    return null;
  }
}

/**
 * Turn a repo listing into ranked download candidates.
 * Split models (…-00001-of-00003.gguf) are collapsed to their first shard —
 * wllama loads the remaining parts itself — instead of being discarded, which
 * is what left large repos with only a poor-quality quant to choose from.
 */
function rankCandidates(files) {
  const singles = [];
  const groups = new Map();

  for (const f of files) {
    const m = f.path.match(SHARD_RE);
    if (!m) { singles.push({ path: f.path, size: f.size, parts: 1 }); continue; }
    const [, base, index, total] = m;
    const g = groups.get(base) ?? { base, size: 0, parts: Number(total), first: null };
    g.size += f.size;
    if (Number(index) === 1) g.first = f.path;
    groups.set(base, g);
  }

  const candidates = [...singles];
  for (const g of groups.values()) {
    if (g.first) candidates.push({ path: g.first, size: g.size, parts: g.parts });
  }

  const rank = (path) => {
    const i = QUANT_ORDER.findIndex((re) => re.test(path));
    return i === -1 ? QUANT_ORDER.length : i;
  };
  return candidates.sort((a, b) => rank(a.path) - rank(b.path) || a.size - b.size);
}

// exported for tests
export { rankCandidates, resolveModelUrl };

/** Cheap existence check: ask for one byte rather than trusting the listing. */
async function urlIsReachable(url) {
  try {
    const res = await fetch(url, { headers: { Range: "bytes=0-0" } });
    return res.ok || res.status === 206;
  } catch {
    return false;
  }
}

/**
 * Resolve a catalog entry to a download URL that is verified to exist.
 * Tries each repo, and within a repo each candidate quant, so a missing or
 * renamed file self-corrects instead of surfacing as a dead link.
 */
async function resolveModelUrl(entry, onStep) {
  for (const repo of entry.repos) {
    onStep?.(`Looking up ${repo}…`);
    const files = await listRepoFiles(repo);
    if (!files || files.length === 0) continue;

    const candidates = rankCandidates(files);
    for (const candidate of candidates.slice(0, 4)) {
      const path = candidate.path.split("/").map(encodeURIComponent).join("/");
      const url = `https://huggingface.co/${repo}/resolve/main/${path}`;
      onStep?.(`Checking ${candidate.path.split("/").pop()}…`);
      if (await urlIsReachable(url)) {
        return { url, repo, file: candidate.path, size: candidate.size, parts: candidate.parts };
      }
    }
  }
  return null;
}

// Answer modes. "Precise" exists to reduce made-up answers as far as sampling
// and prompting can: greedy decoding (no dice rolls), and explicit permission
// to say "I don't know". It cannot eliminate hallucination — no model can, and
// small ones least of all — but it is a large, measurable improvement over
// creative sampling.
const MODES = {
  precise: {
    label: "Precise",
    temperature: 0,
    system:
      "You are PocketGPT, running entirely on the user's phone. " +
      "Answer only what you actually know. If you are unsure, or the question " +
      "needs information you do not have, say so plainly instead of guessing. " +
      "Never invent facts, numbers, dates, names, quotations, citations or links. " +
      "If you are estimating, say that it is an estimate. Keep answers short and direct.",
  },
  balanced: {
    label: "Balanced",
    temperature: 0.6,
    system:
      "You are PocketGPT, a helpful assistant running entirely on the user's phone. " +
      "Answer clearly and concisely. If you are unsure about something, say so.",
  },
  creative: {
    label: "Creative",
    temperature: 0.95,
    system:
      "You are PocketGPT, a helpful and imaginative assistant running entirely " +
      "on the user's phone. Be expressive and willing to explore ideas.",
  },
};

const DEFAULT_SYSTEM = MODES.precise.system;

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem("pgpt." + k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("pgpt." + k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem("pgpt." + k); } catch {} },
};

let wllama = null;
let messages = [];
let abortController = null;
let generating = false;
let isQwen3 = false;

/** Worker threads only help when the page is cross-origin isolated (Turbo). */
function threadCount() {
  if (!self.crossOriginIsolated) return 1;
  const cores = navigator.hardwareConcurrency || 4;
  // leave a core for the UI; more threads than physical cores hurts
  return Math.max(1, Math.min(8, cores - 1));
}

const settings = {
  mode: store.get("mode", "precise"),
  maxTokens: store.get("maxTokens", 512),
  contextLength: store.get("contextLength", 4096),
  turbo: store.get("turbo", false),
  system: store.get("system", DEFAULT_SYSTEM),
};

// shown in Settings so it is obvious whether a deploy has actually landed
const BUILD = "2026-07-25.7";

const modeConfig = () => MODES[settings.mode] ?? MODES.precise;

// ---------- screens ----------

function show(screen) {
  const chatOn = screen === "chat";
  $("setup").classList.toggle("hidden", screen !== "setup");
  $("progress").classList.toggle("hidden", screen !== "progress");
  $("chat").classList.toggle("hidden", !chatOn);
  $("composer").classList.toggle("hidden", !chatOn);
  $("stats").classList.toggle("hidden", !chatOn);
  $("newchat").classList.toggle("hidden", !chatOn);
  $("settings").classList.toggle("hidden", !chatOn);
}

function renderCatalog() {
  const host = $("models");
  host.innerHTML = "";
  for (const m of CATALOG) {
    const btn = document.createElement("button");
    btn.className = "model";
    btn.innerHTML =
      `<div class="top"><span class="name"></span><span class="size"></span></div>
       <div class="note"></div>`;
    btn.querySelector(".name").textContent = m.name;
    if (m.recommended) {
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = "recommended";
      btn.querySelector(".name").appendChild(b);
    }
    btn.querySelector(".size").textContent = m.size;
    btn.querySelector(".note").textContent = m.note;
    btn.onclick = () => loadCatalogModel(m);
    host.appendChild(btn);
  }
}

function showSetupError(text) {
  const box = $("setuperror");
  box.textContent = text;
  box.classList.remove("hidden");
  show("setup");
}

function clearSetupError() {
  $("setuperror").classList.add("hidden");
}

function setProgress(fraction, text) {
  $("barfill").style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  $("ptext").textContent = text;
}

// ---------- model loading ----------

/** Resolve a catalog entry against Hugging Face, then load whatever exists. */
async function loadCatalogModel(entry) {
  clearSetupError();
  show("progress");
  setProgress(0, `Finding ${entry.name}…`);
  try {
    const found = await resolveModelUrl(entry, (step) => setProgress(0, step));
    if (!found) {
      showSetupError(
        `Could not find a download for ${entry.name}. The repositories may be ` +
        `offline or require sign-in. Try another model, or paste a direct ` +
        `.gguf URL below.`);
      return;
    }
    await loadModel(found.url, entry.name);
  } catch (err) {
    showSetupError(`Could not load ${entry.name}: ${err?.message ?? err}`);
  }
}

async function loadModel(url, label) {
  show("progress");
  setProgress(0, `Preparing ${label}…`);
  abortController = new AbortController();

  try {
    if (wllama) { try { await wllama.exit(); } catch {} }
    wllama = new Wllama(WASM_PATHS, { allowOffline: true, suppressNativeLog: true });

    let lastPct = -1;
    const common = {
      n_ctx: settings.contextLength,
      progressCallback: ({ loaded, total }) => {
        if (!total) return;
        const pct = Math.floor((loaded / total) * 100);
        if (pct === lastPct) return;
        lastPct = pct;
        const mb = (n) => (n / 1e6).toFixed(0);
        setProgress(loaded / total,
          `Downloading ${label}\n${mb(loaded)} / ${mb(total)} MB — ${pct}%`);
      },
      signal: abortController.signal,
    };

    // Performance settings, each measured against this WebAssembly build:
    //  - n_threads  : verified OK. No-op unless the page is cross-origin
    //                 isolated (Turbo), where it gives real multi-core decode.
    //  - flash_attn : verified OK. Faster attention, less memory per token.
    //  - quantised KV cache (cache_type_k/v = q8_0): DELIBERATELY OMITTED — it
    //    would halve KV memory, but it hangs the loader here, and a hang is not
    //    something the fallback below can rescue.
    const tuned = { ...common, n_threads: threadCount(), flash_attn: true };

    try {
      await wllama.loadModelFromUrl(url, tuned);
    } catch (err) {
      if (err?.name === "AbortError") throw err;
      console.warn("tuned load failed, retrying with defaults:", err);
      setProgress(1, "Starting the model…");
      await wllama.loadModelFromUrl(url, common);
    }

    setProgress(1, "Starting the model…");
    store.set("model", { url, label });
    isQwen3 = /qwen3/i.test(`${label} ${url}`);

    let meta = "";
    try {
      const m = wllama.getModelMetadata();
      const h = m.hparams;
      meta = `${label}\n${h.nLayer} layers · ${h.nEmbd} hidden · vocab ${h.nVocab}`;
    } catch { meta = label; }
    $("modelinfo").textContent = meta;
    $("sub").textContent = `${label} · on-device`;

    startNewChat();
    show("chat");
    $("box").focus();
  } catch (err) {
    if (err?.name === "AbortError") { show("setup"); return; }
    store.del("model");   // don't retry a broken model on every launch
    showSetupError(`Could not load ${label}: ${err?.message ?? err}`);
  }
}

async function resumeSavedModel() {
  const saved = store.get("model", null);
  if (!saved) { show("setup"); return; }
  // cached models load from the browser's storage, so this is usually instant
  loadModel(saved.url, saved.label);
}

// ---------- chat ----------

function addBubble(cls, text) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

function startNewChat() {
  abortGeneration();
  messages = [{ role: "system", content: settings.system }];
  $("modelabel").textContent = modeConfig().label;
  $("chat").innerHTML = "";
  addBubble("note", "Everything below is computed on your phone. Offline works.");
  $("stats").textContent = "";
}

function setGenerating(on) {
  generating = on;
  $("send").textContent = on ? "■" : "↑";
  $("send").classList.toggle("stop", on);
}

function abortGeneration() {
  if (abortController && generating) abortController.abort();
  setGenerating(false);
}

async function sendMessage() {
  if (generating) { abortGeneration(); return; }
  const text = $("box").value.trim();
  if (!text || !wllama) return;

  $("box").value = "";
  $("box").style.height = "auto";
  addBubble("user", text);
  messages.push({ role: "user", content: text });

  const bubble = addBubble("bot", "…");
  setGenerating(true);
  abortController = new AbortController();

  let reply = "";
  let tokens = 0;
  let failure = null;
  const t0 = performance.now();

  // A full context window is normal in a long chat: drop the oldest turns
  // (never the system prompt) and try again.
  for (let attempt = 0; attempt < 4; attempt++) {
    failure = null;
    try {
      await wllama.createChatCompletion({
        messages,
        stream: true,
        temperature: modeConfig().temperature,
        max_tokens: settings.maxTokens,
        // reuse the KV cache for the conversation prefix instead of
        // reprocessing every earlier turn — the single biggest saving in a
        // multi-turn chat, in both tokens and time
        cache_prompt: true,
        // Qwen3 emits a long private "thinking" section by default; that is
        // pure token cost for a phone chat, so turn it off where supported
        ...(isQwen3 ? { chat_template_kwargs: { enable_thinking: false } } : {}),
        abortSignal: abortController.signal,
        onData: (chunk) => {
          const piece = chunk?.choices?.[0]?.delta?.content;
          if (!piece) return;
          reply += piece;
          tokens++;
          bubble.textContent = reply;
          $("chat").scrollTop = $("chat").scrollHeight;
          const secs = (performance.now() - t0) / 1000;
          if (secs > 0.4) $("stats").textContent = `${(tokens / secs).toFixed(1)} tok/s`;
        },
      });
      break;
    } catch (err) {
      failure = err;
      if (err?.name === "AbortError") break;
      if (isContextOverflow(err) && dropOldestTurn()) {
        reply = "";
        tokens = 0;
        bubble.textContent = "…";
        continue;
      }
      break;
    }
  }

  if (failure && failure.name !== "AbortError" && !reply) {
    bubble.textContent = `⚠️ ${describeError(failure)}`;
  } else if (!reply.trim()) {
    bubble.textContent = "(no reply)";
  }
  messages.push({ role: "assistant", content: reply });
  setGenerating(false);
}

function isContextOverflow(err) {
  const text = `${err?.message ?? ""} ${err?.type ?? ""}`.toLowerCase();
  return text.includes("context size") || text.includes("exceed_context");
}

/** Remove the oldest user/assistant exchange, keeping the system prompt. */
function dropOldestTurn() {
  const first = messages.findIndex((m) => m.role !== "system");
  // need at least the current user message left after trimming
  if (first < 0 || messages.length - first <= 1) return false;
  messages.splice(first, messages[first + 1]?.role === "assistant" ? 2 : 1);
  return true;
}

function describeError(err) {
  const message = err?.message ?? String(err);
  if (isContextOverflow(err)) {
    return "This conversation no longer fits in the model's context. Tap New to start a fresh chat.";
  }
  return message;
}

// ---------- settings ----------

function renderModeButtons() {
  const host = $("modes");
  host.innerHTML = "";
  for (const [key, cfg] of Object.entries(MODES)) {
    const b = document.createElement("button");
    b.className = "modebtn" + (key === settings.mode ? " active" : "");
    b.textContent = cfg.label;
    b.onclick = () => {
      settings.mode = key;
      // the system prompt follows the mode unless it has been hand-edited
      if (!store.get("systemEdited", false)) {
        settings.system = cfg.system;
        $("sysprompt").value = cfg.system;
      }
      renderModeButtons();
    };
    host.appendChild(b);
  }
}

function openSheet() {
  renderModeButtons();
  $("maxtok").value = settings.maxTokens;
  $("maxtokv").textContent = settings.maxTokens;
  $("ctxlen").value = String(settings.contextLength);
  $("turbo").checked = settings.turbo;
  $("turbostate").textContent = self.crossOriginIsolated
    ? `active · ${threadCount()} threads` : "off · single core";
  $("sysprompt").value = settings.system;
  $("build").textContent = `build ${BUILD}`;
  $("sheet").classList.remove("hidden");
}

function closeSheet() {
  settings.maxTokens = Number($("maxtok").value);
  const sys = $("sysprompt").value.trim() || modeConfig().system;
  const systemChanged = sys !== settings.system;
  settings.system = sys;
  if (sys !== modeConfig().system) store.set("systemEdited", true);

  const newCtx = Number($("ctxlen").value);
  const ctxChanged = newCtx !== settings.contextLength;
  settings.contextLength = newCtx;

  const newTurbo = $("turbo").checked;
  const turboChanged = newTurbo !== settings.turbo;
  settings.turbo = newTurbo;

  store.set("mode", settings.mode);
  store.set("maxTokens", settings.maxTokens);
  store.set("contextLength", settings.contextLength);
  store.set("turbo", settings.turbo);
  store.set("system", settings.system);
  if (systemChanged && messages.length) messages[0] = { role: "system", content: sys };
  $("modelabel").textContent = modeConfig().label;
  $("sheet").classList.add("hidden");

  // Turbo changes how the page itself is served, so it needs a reload
  if (turboChanged) { applyTurbo().then(() => location.reload()); return; }

  // context length is fixed when the model is created, so reload it (the file
  // is already stored locally, so this is quick)
  if (ctxChanged) {
    const saved = store.get("model", null);
    if (saved) loadModel(saved.url, saved.label);
  }
}

/**
 * Turbo = cross-origin isolation, which is what unlocks SharedArrayBuffer and
 * therefore multi-threaded inference. GitHub Pages cannot send the required
 * headers, so the service worker adds them to our own responses instead.
 */
async function applyTurbo() {
  if (!("serviceWorker" in navigator)) return;
  try {
    const regs = await navigator.serviceWorker.getRegistrations();
    await Promise.all(regs.map((r) => r.unregister()));
    await navigator.serviceWorker.register(settings.turbo ? "sw.js?coi=1" : "sw.js");
  } catch {}
}

// ---------- wiring ----------

$("send").onclick = sendMessage;
$("box").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$("box").addEventListener("input", () => {
  $("box").style.height = "auto";
  $("box").style.height = Math.min($("box").scrollHeight, 110) + "px";
});
$("newchat").onclick = startNewChat;
$("settings").onclick = openSheet;
$("closesheet").onclick = closeSheet;
$("maxtok").oninput = () => { $("maxtokv").textContent = $("maxtok").value; };
$("switchmodel").onclick = async () => {
  closeSheet();
  abortGeneration();
  if (wllama) { try { await wllama.exit(); } catch {} }
  wllama = null;
  store.del("model");
  show("setup");
};
$("cancel").onclick = () => { abortController?.abort(); show("setup"); };
$("customgo").onclick = () => {
  const url = $("customurl").value.trim();
  if (!url) return;
  loadModel(url, url.split("/").pop() || "custom model");
};

// Register the service worker and adopt new versions automatically: the new
// worker calls skipWaiting()/claim(), which fires controllerchange, and we
// reload once so the freshly deployed app is what the user is looking at.
if ("serviceWorker" in navigator) {
  let reloading = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloading) return;
    reloading = true;
    location.reload();
  });
  navigator.serviceWorker.register(settings.turbo ? "sw.js?coi=1" : "sw.js")
    .then((reg) => reg.update().catch(() => {}))
    .catch(() => {});
}

renderCatalog();
resumeSavedModel();
