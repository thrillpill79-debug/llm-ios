// PocketGPT — a real instruct-tuned LLM running fully in the browser via
// llama.cpp compiled to WebAssembly (wllama). Models are downloaded once
// from Hugging Face, stored in the browser's origin-private filesystem, and
// run entirely on-device: no server, no API key, no data leaving the phone.
import { Wllama } from "./vendor/wllama/index.js";

const WASM_PATHS = { default: "./vendor/wllama/wllama.wasm" };

// Sized for a phone browser tab, which gets a much smaller memory budget than
// a native app. 0.5B-class models are the reliable choice on iPhone.
const CATALOG = [
  {
    name: "Qwen2.5 0.5B Instruct",
    size: "~400 MB",
    recommended: true,
    note: "Best fit for iPhone. Real assistant behaviour, answers in a few seconds.",
    url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
  },
  {
    name: "SmolLM2 360M Instruct",
    size: "~270 MB",
    note: "Smallest and fastest. Chattier than its size suggests, but weak at facts.",
    url: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q4_k_m.gguf",
  },
  {
    name: "Llama 3.2 1B Instruct",
    size: "~810 MB",
    note: "Noticeably smarter. Heavy for a phone tab — may reload the page on older devices.",
    url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  },
];

const DEFAULT_SYSTEM =
  "You are PocketGPT, a helpful assistant running entirely on the user's phone. " +
  "Answer clearly and concisely.";

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

const settings = {
  temperature: store.get("temperature", 0.7),
  maxTokens: store.get("maxTokens", 512),
  system: store.get("system", DEFAULT_SYSTEM),
};

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
    btn.onclick = () => loadModel(m.url, m.name);
    host.appendChild(btn);
  }
}

function setProgress(fraction, text) {
  $("barfill").style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  $("ptext").textContent = text;
}

// ---------- model loading ----------

async function loadModel(url, label) {
  show("progress");
  setProgress(0, `Preparing ${label}…`);
  abortController = new AbortController();

  try {
    if (wllama) { try { await wllama.exit(); } catch {} }
    wllama = new Wllama(WASM_PATHS, { allowOffline: true, suppressNativeLog: true });

    let lastPct = -1;
    await wllama.loadModelFromUrl(url, {
      n_ctx: 2048,
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
    });

    setProgress(1, "Starting the model…");
    store.set("model", { url, label });

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
    setProgress(0, `Could not load the model.\n${err?.message ?? err}`);
    setTimeout(() => show("setup"), 2600);
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
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
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

function openSheet() {
  $("temp").value = settings.temperature;
  $("tempv").textContent = Number(settings.temperature).toFixed(2);
  $("maxtok").value = settings.maxTokens;
  $("maxtokv").textContent = settings.maxTokens;
  $("sysprompt").value = settings.system;
  $("sheet").classList.remove("hidden");
}

function closeSheet() {
  settings.temperature = Number($("temp").value);
  settings.maxTokens = Number($("maxtok").value);
  const sys = $("sysprompt").value.trim() || DEFAULT_SYSTEM;
  const systemChanged = sys !== settings.system;
  settings.system = sys;
  store.set("temperature", settings.temperature);
  store.set("maxTokens", settings.maxTokens);
  store.set("system", settings.system);
  if (systemChanged && messages.length) messages[0] = { role: "system", content: sys };
  $("sheet").classList.add("hidden");
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
$("temp").oninput = () => { $("tempv").textContent = Number($("temp").value).toFixed(2); };
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

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

renderCatalog();
resumeSavedModel();
