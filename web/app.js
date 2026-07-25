import { TinyGPT, CharTokenizer, sliceWeights, sample } from "./tinygpt.js";

const $ = (id) => document.getElementById(id);
const out = $("out"), go = $("go"), promptEl = $("prompt"), status = $("status");

$("temp").oninput = () => { $("tempv").textContent = Number($("temp").value).toFixed(2); };
$("ntok").oninput = () => { $("ntokv").textContent = $("ntok").value; };

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

let model, tokenizer, running = false;

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
  $("modelinfo").textContent =
    `${(params / 1000).toFixed(0)}k params · on-device`;
  go.disabled = false;
}

async function generate() {
  if (running) { running = false; return; }
  running = true;
  go.textContent = "Stop";
  go.classList.add("stop");
  const prompt = promptEl.value;
  const temperature = Number($("temp").value);
  const maxTokens = Number($("ntok").value);

  out.classList.remove("empty");
  out.textContent = prompt;
  model.reset();

  let ids = tokenizer.encode(prompt);
  if (ids.length === 0) ids = [0];
  let logits;
  for (const id of ids) logits = model.step(id);

  const t0 = performance.now();
  let generated = 0;
  while (running && generated < maxTokens) {
    // a few tokens per frame keeps the UI at full responsiveness
    for (let burst = 0; burst < 4 && generated < maxTokens; burst++) {
      const next = sample(logits, temperature, 40);
      out.textContent += tokenizer.decode([next]);
      logits = model.step(next);
      generated++;
    }
    out.scrollTop = out.scrollHeight;
    const tps = generated / ((performance.now() - t0) / 1000);
    status.textContent = `${generated} tokens · ${tps.toFixed(0)} tok/s`;
    await new Promise(requestAnimationFrame);
  }

  running = false;
  go.textContent = "Generate";
  go.classList.remove("stop");
}

go.onclick = generate;
promptEl.onkeydown = (e) => {
  if (e.key === "Enter" && !running && !go.disabled) { e.preventDefault(); generate(); }
};

loadModel().catch((err) => {
  status.textContent = `Failed to load model: ${err.message}`;
});
