// Chat UI over the TinyLLM engine. The model is a char-level Shakespeare
// model, so the conversation is staged as a play script — the user speaks
// as one character, the model replies as another. Generation stops when the
// model starts the next speaker heading (or hits the token cap).
import { TinyGPT, CharTokenizer, sliceWeights, sample } from "./tinygpt.js";

const $ = (id) => document.getElementById(id);
const chat = $("chat"), box = $("box"), send = $("send"), status = $("status");

const USER_NAME = "ROMEO";   // the user's speaker heading in the script
const BOT_NAME = "JULIET";   // the model's speaker heading
const MAX_REPLY_TOKENS = 220;

$("temp").oninput = () => { $("tempv").textContent = Number($("temp").value).toFixed(2); };

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

let model, tokenizer, generating = false;
let script = "";  // the running play-script transcript fed to the model

function addBubble(cls, text) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function loadModel() {
  const [config, vocab, manifest] = await Promise.all([
    fetch("model/config.json").then(r => r.json()),
    fetch("model/vocab.json").then(r => r.json()),
    fetch("model/manifest.json").then(r => r.json()),
  ]);
  const buffer = await fetch("model/weights.bin").then(r => r.arrayBuffer());
  tokenizer = new CharTokenizer(vocab.chars);
  model = new TinyGPT(config, sliceWeights(manifest, buffer));
  const params = manifest.reduce((s, e) => s + e.count, 0);
  $("modelinfo").textContent = `${(params / 1000).toFixed(0)}k params · on-device`;
  send.disabled = false;
}

// a reply ends when the model begins another "SPEAKER:" heading after a
// blank line — trim it (and trailing whitespace) off
function trimReply(text) {
  const m = text.match(/\n\s*\n[A-Z][A-Za-z' ]{1,20}:/);
  return (m ? text.slice(0, m.index) : text).replace(/\s+$/, "");
}

async function respond() {
  const userText = box.value.trim();
  if (!userText || generating || !model) return;
  generating = true;
  send.disabled = true;
  box.value = "";
  addBubble("user", userText);

  script += `${USER_NAME}:\n${userText}\n\n${BOT_NAME}:\n`;
  const bubble = addBubble("bot", "…");

  // re-prefill the transcript (the engine keeps only the last block_size
  // chars of context internally — tiny model, tiny memory)
  model.reset();
  let ids = tokenizer.encode(script);
  if (ids.length === 0) ids = [0];
  let logits;
  for (const id of ids) logits = model.step(id);

  const temperature = Number($("temp").value);
  const t0 = performance.now();
  let reply = "";
  let tokens = 0;
  while (tokens < MAX_REPLY_TOKENS) {
    for (let burst = 0; burst < 4 && tokens < MAX_REPLY_TOKENS; burst++) {
      const next = sample(logits, temperature, 40);
      reply += tokenizer.decode([next]);
      logits = model.step(next);
      tokens++;
    }
    const shown = trimReply(reply);
    bubble.textContent = shown.length ? shown : "…";
    chat.scrollTop = chat.scrollHeight;
    const tps = tokens / ((performance.now() - t0) / 1000);
    status.textContent = `${tps.toFixed(0)} tok/s · context: last ${model.cfg.block_size} chars`;
    // stop as soon as the model starts the next speaker heading
    if (trimReply(reply) !== reply.replace(/\s+$/, "")) break;
    await new Promise(requestAnimationFrame);
  }

  const finalReply = trimReply(reply) || "(silence)";
  bubble.textContent = finalReply;
  script += `${finalReply}\n\n`;
  chat.scrollTop = chat.scrollHeight;
  generating = false;
  send.disabled = false;
}

send.onclick = respond;
box.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); respond(); } };

loadModel().catch((err) => {
  status.textContent = `Failed to load model: ${err.message}`;
});
