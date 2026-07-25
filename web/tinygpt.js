// TinyGPT inference in pure JavaScript — the third sibling of
// python/verify_export.py (NumPy) and ios/TinyLLM/Engine/TinyGPT.swift.
// Loads the same exported model files and produces the same logits.
// KV-cached single-token stepping; no dependencies.

// Abramowitz & Stegun 7.1.26 erf approximation (max error 1.5e-7),
// close enough to the exact-erf GELU used in training
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}

const SQRT2 = Math.sqrt(2);

function gelu(x, n) {
  for (let i = 0; i < n; i++) x[i] = 0.5 * x[i] * (1 + erf(x[i] / SQRT2));
}

function layerNorm(x, n, gain, bias) {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; variance += d * d; }
  variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < n; i++) x[i] = (x[i] - mean) * inv * gain[i] + bias[i];
}

function softmaxInPlace(x, n) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = 0; i < n; i++) { x[i] = Math.exp(x[i] - max); sum += x[i]; }
  for (let i = 0; i < n; i++) x[i] /= sum;
}

// out[r] = sum_c w[r*cols+c] * x[c] (+ bias[r]);  w is [rows x cols] row-major
function matVec(w, rows, cols, x, bias, out) {
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) acc += w[base + c] * x[c];
    out[r] = bias ? acc + bias[r] : acc;
  }
}

export class CharTokenizer {
  constructor(chars) {
    this.chars = chars;
    this.stoi = new Map(chars.map((c, i) => [c, i]));
  }
  encode(text) {
    const ids = [];
    for (const ch of text) {
      const id = this.stoi.get(ch);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }
  decode(ids) { return ids.map(i => this.chars[i]).join(""); }
}

export class TinyGPT {
  constructor(config, weights) {
    this.cfg = config;
    this.w = weights; // {name: Float32Array}
    this.headDim = config.n_embd / config.n_head;
    const E = config.n_embd;
    // scratch buffers
    this.x = new Float32Array(E);
    this.h = new Float32Array(E);
    this.qkv = new Float32Array(3 * E);
    this.attnOut = new Float32Array(E);
    this.proj = new Float32Array(E);
    this.mid = new Float32Array(4 * E);
    this.scores = new Float32Array(config.block_size);
    this.logits = new Float32Array(config.vocab_size);
    // per-layer KV caches: [block_size x E], head h of position t at t*E + h*hd
    this.kCache = [];
    this.vCache = [];
    for (let l = 0; l < config.n_layer; l++) {
      this.kCache.push(new Float32Array(config.block_size * E));
      this.vCache.push(new Float32Array(config.block_size * E));
    }
    this.cached = [];
  }

  reset() { this.cached.length = 0; }

  // feed one token, return logits for the next (Float32Array, reused!)
  step(token) {
    if (this.cached.length === this.cfg.block_size) {
      // context full: re-prefill from the most recent half of the window
      const tail = this.cached.slice(-this.cfg.block_size / 2);
      this.reset();
      for (const t of tail) this._forwardOne(t);
    }
    return this._forwardOne(token);
  }

  _forwardOne(token) {
    const { cfg, w, headDim } = this;
    const E = cfg.n_embd;
    const pos = this.cached.length;
    const { x, h, qkv, attnOut, proj, mid, scores } = this;

    const tokEmb = w["tok_emb.weight"];
    const posEmb = w["pos_emb.weight"];
    for (let i = 0; i < E; i++) x[i] = tokEmb[token * E + i] + posEmb[pos * E + i];

    for (let l = 0; l < cfg.n_layer; l++) {
      const p = `blocks.${l}.`;

      // causal self-attention over the cache
      h.set(x);
      layerNorm(h, E, w[p + "ln1.weight"], w[p + "ln1.bias"]);
      matVec(w[p + "attn.qkv.weight"], 3 * E, E, h, w[p + "attn.qkv.bias"], qkv);
      const kC = this.kCache[l], vC = this.vCache[l];
      kC.set(qkv.subarray(E, 2 * E), pos * E);
      vC.set(qkv.subarray(2 * E, 3 * E), pos * E);
      const T = pos + 1;
      const scale = 1 / Math.sqrt(headDim);

      attnOut.fill(0);
      for (let head = 0; head < cfg.n_head; head++) {
        const hOff = head * headDim;
        for (let t = 0; t < T; t++) {
          let acc = 0;
          const kBase = t * E + hOff;
          for (let d = 0; d < headDim; d++) acc += qkv[hOff + d] * kC[kBase + d];
          scores[t] = acc * scale;
        }
        softmaxInPlace(scores, T);
        for (let t = 0; t < T; t++) {
          const prob = scores[t];
          const vBase = t * E + hOff;
          for (let d = 0; d < headDim; d++) attnOut[hOff + d] += prob * vC[vBase + d];
        }
      }
      matVec(w[p + "attn.proj.weight"], E, E, attnOut, w[p + "attn.proj.bias"], proj);
      for (let i = 0; i < E; i++) x[i] += proj[i];

      // MLP
      h.set(x);
      layerNorm(h, E, w[p + "ln2.weight"], w[p + "ln2.bias"]);
      matVec(w[p + "mlp.fc.weight"], 4 * E, E, h, w[p + "mlp.fc.bias"], mid);
      gelu(mid, 4 * E);
      matVec(w[p + "mlp.proj.weight"], E, 4 * E, mid, w[p + "mlp.proj.bias"], proj);
      for (let i = 0; i < E; i++) x[i] += proj[i];
    }

    layerNorm(x, E, w["ln_f.weight"], w["ln_f.bias"]);
    matVec(tokEmb, cfg.vocab_size, E, x, null, this.logits); // weight-tied head

    this.cached.push(token);
    return this.logits;
  }
}

export function sample(logits, temperature, topK, rng = Math.random) {
  const n = logits.length;
  const l = Float32Array.from(logits);
  if (temperature <= 0.05) {
    let best = 0;
    for (let i = 1; i < n; i++) if (l[i] > l[best]) best = i;
    return best;
  }
  for (let i = 0; i < n; i++) l[i] /= temperature;
  if (topK > 0 && topK < n) {
    const sorted = Float32Array.from(l).sort().reverse();
    const threshold = sorted[topK - 1];
    for (let i = 0; i < n; i++) if (l[i] < threshold) l[i] = -Infinity;
  }
  softmaxInPlace(l, n);
  let r = rng();
  for (let i = 0; i < n; i++) {
    r -= l[i];
    if (r < 0) return i;
  }
  return n - 1;
}

// parse weights.bin + manifest into named Float32Array views
export function sliceWeights(manifest, buffer) {
  const all = new Float32Array(buffer);
  const weights = {};
  for (const e of manifest) {
    weights[e.name] = all.subarray(e.offset, e.offset + e.count);
  }
  return weights;
}
