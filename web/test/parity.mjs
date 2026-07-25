// Node parity test for the JS engine: same reference values as
// swift-tests/TinyGPTParityTests.swift (computed from the NumPy reference,
// which is verified against the PyTorch checkpoint).
//   node web/test/parity.mjs   (from the repo root)
import { readFileSync } from "node:fs";
import { TinyGPT, CharTokenizer, sliceWeights } from "../tinygpt.js";

const dir = process.env.LLM_EXPORT_DIR ?? "model/export";
const config = JSON.parse(readFileSync(`${dir}/config.json`, "utf8"));
const vocab = JSON.parse(readFileSync(`${dir}/vocab.json`, "utf8"));
const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, "utf8"));
const bin = readFileSync(`${dir}/weights.bin`);
const weights = sliceWeights(
  manifest,
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
);

const PROMPT_IDS = [30, 27, 25, 17, 27, 10]; // "ROMEO:"
const EXPECTED_LOGITS = [
  11.46425, 5.41142, -0.03004, -10.41260, -8.46380, 1.88023, 0.71950,
  1.86731, 1.63704, -0.50594, 2.19826, 0.59514, 0.53068, -1.21386,
  -0.92888, -1.20915, -1.73049, -1.42563, -1.60739, -1.29741, -2.04260,
  -2.38479, -3.51607, -2.59296, -0.79570, -0.04534, -1.27789, -0.12205,
  -1.01678, -2.81154, -1.35241, 0.51171, -0.10653, -2.39690, -1.65711,
  -0.56555, -5.20546, -1.28375, -2.70574, -0.71421, -2.49737, -1.25613,
  -0.80969, -1.46447, -1.76870, -2.24076, -2.42797, -1.28313, -3.31727,
  -3.42108, -0.24413, -0.89051, -0.37645, -1.37398, -0.72607, -3.46059,
  -0.86810, 0.38661, -1.16513, -0.90878, -1.76574, -1.10586, -1.72257,
  -1.29914, -3.77979,
];
const EXPECTED_GREEDY = [0, 21, 1, 61, 53, 59, 50, 42, 1, 58,
                        46, 43, 1, 57, 46, 39, 50, 50, 1, 58];

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
  if (!ok) failures++;
};

const tokenizer = new CharTokenizer(vocab.chars);
check("tokenizer encode", JSON.stringify(tokenizer.encode("ROMEO:")) === JSON.stringify(PROMPT_IDS));
check("tokenizer decode", tokenizer.decode(PROMPT_IDS) === "ROMEO:");

const model = new TinyGPT(config, weights);
let logits;
for (const id of PROMPT_IDS) logits = model.step(id);
let worst = 0;
for (let i = 0; i < EXPECTED_LOGITS.length; i++) {
  worst = Math.max(worst, Math.abs(logits[i] - EXPECTED_LOGITS[i]));
}
check("logits vs reference", worst < 5e-3, `max diff ${worst.toExponential(2)}`);

const produced = [];
for (let s = 0; s < EXPECTED_GREEDY.length; s++) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  produced.push(best);
  logits = model.step(best);
}
check("greedy generation", JSON.stringify(produced) === JSON.stringify(EXPECTED_GREEDY),
      tokenizer.decode(produced).replace(/\n/g, "\\n"));

// overflow path: run past block_size, logits must stay finite
model.reset();
logits = model.step(0);
for (let i = 0; i < config.block_size + 40; i++) {
  logits = model.step(i % config.vocab_size);
}
check("context overflow re-prefill", [...logits].every(Number.isFinite)
      && model.cached.length <= config.block_size);

process.exit(failures ? 1 : 0);
