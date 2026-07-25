// On-device inference for the GPT trained by python/train.py.
//
// The math mirrors python/verify_export.py line by line, but processes one
// token at a time with a per-layer KV cache, so each generated token costs a
// single forward pass over the new position only. When the context window
// fills up, the cache is rebuilt from the most recent half of the window
// (learned absolute positions make cached entries position-dependent, so a
// plain sliding window would be wrong).

import Foundation

// @unchecked Sendable: the generation pipeline touches a TinyGPT from one
// task at a time (Generator serializes access); the annotation keeps the
// engine usable from a background Task under Swift 6 strict concurrency.
final class TinyGPT: @unchecked Sendable {
    let config: ModelConfig
    private let w: [String: [Float]]
    private let headDim: Int

    // kCache[layer] / vCache[layer] hold one n_embd-length k/v vector per
    // cached position, concatenated; head h of position t lives at
    // [t * n_embd + h * headDim ..< t * n_embd + (h+1) * headDim].
    private var kCache: [[Float]]
    private var vCache: [[Float]]
    private(set) var cachedTokens: [Int] = []

    init(checkpoint: Checkpoint) {
        self.config = checkpoint.config
        self.w = checkpoint.tensors
        self.headDim = config.n_embd / config.n_head
        self.kCache = Array(repeating: [], count: config.n_layer)
        self.vCache = Array(repeating: [], count: config.n_layer)
    }

    func reset() {
        for i in 0..<config.n_layer {
            kCache[i].removeAll(keepingCapacity: true)
            vCache[i].removeAll(keepingCapacity: true)
        }
        cachedTokens.removeAll(keepingCapacity: true)
    }

    /// Feed one token; returns the logits for the next token.
    func step(_ token: Int) -> [Float] {
        if cachedTokens.count == config.block_size {
            // context full: re-prefill from the most recent half of the window
            let tail = Array(cachedTokens.suffix(config.block_size / 2))
            reset()
            for t in tail { _ = forwardOne(t) }
        }
        return forwardOne(token)
    }

    private func forwardOne(_ token: Int) -> [Float] {
        let E = config.n_embd
        let pos = cachedTokens.count

        var x = [Float](repeating: 0, count: E)
        let tokEmb = w["tok_emb.weight"]!
        let posEmb = w["pos_emb.weight"]!
        for i in 0..<E {
            x[i] = tokEmb[token * E + i] + posEmb[pos * E + i]
        }

        for layer in 0..<config.n_layer {
            let p = "blocks.\(layer)."

            // -- causal self-attention over the cache --
            var h = x
            MathOps.layerNorm(&h, gain: w[p + "ln1.weight"]!, bias: w[p + "ln1.bias"]!)
            let qkv = MathOps.matVec(w[p + "attn.qkv.weight"]!, rows: 3 * E, cols: E,
                                     h, bias: w[p + "attn.qkv.bias"]!)
            let q = Array(qkv[0..<E])
            kCache[layer].append(contentsOf: qkv[E..<(2 * E)])
            vCache[layer].append(contentsOf: qkv[(2 * E)..<(3 * E)])
            let T = pos + 1

            var attnOut = [Float](repeating: 0, count: E)
            let scale = 1.0 / sqrt(Float(headDim))
            q.withUnsafeBufferPointer { qp in
                kCache[layer].withUnsafeBufferPointer { kp in
                    vCache[layer].withUnsafeBufferPointer { vp in
                        var scores = [Float](repeating: 0, count: T)
                        for head in 0..<config.n_head {
                            let hOff = head * headDim
                            for t in 0..<T {
                                scores[t] = MathOps.dot(qp.baseAddress! + hOff,
                                                        kp.baseAddress! + t * E + hOff,
                                                        headDim) * scale
                            }
                            MathOps.softmax(&scores)
                            for t in 0..<T {
                                let prob = scores[t]
                                let vBase = t * E + hOff
                                for d in 0..<headDim {
                                    attnOut[hOff + d] += prob * vp[vBase + d]
                                }
                            }
                        }
                    }
                }
            }
            let attnProj = MathOps.matVec(w[p + "attn.proj.weight"]!, rows: E, cols: E,
                                          attnOut, bias: w[p + "attn.proj.bias"]!)
            for i in 0..<E { x[i] += attnProj[i] }

            // -- MLP --
            h = x
            MathOps.layerNorm(&h, gain: w[p + "ln2.weight"]!, bias: w[p + "ln2.bias"]!)
            var mid = MathOps.matVec(w[p + "mlp.fc.weight"]!, rows: 4 * E, cols: E,
                                     h, bias: w[p + "mlp.fc.bias"]!)
            MathOps.gelu(&mid)
            let mlpOut = MathOps.matVec(w[p + "mlp.proj.weight"]!, rows: E, cols: 4 * E,
                                        mid, bias: w[p + "mlp.proj.bias"]!)
            for i in 0..<E { x[i] += mlpOut[i] }
        }

        MathOps.layerNorm(&x, gain: w["ln_f.weight"]!, bias: w["ln_f.bias"]!)
        // weight-tied LM head: logits = tok_emb · x
        let logits = MathOps.matVec(tokEmb, rows: config.vocab_size, cols: E, x)

        cachedTokens.append(token)
        return logits
    }
}

// MARK: - Sampling

struct Sampler {
    var temperature: Float = 0.8
    var topK: Int = 40

    func sample(_ logits: [Float]) -> Int {
        var l = logits
        let temp = max(temperature, 1e-6)
        for i in 0..<l.count { l[i] /= temp }

        if topK > 0 && topK < l.count {
            let threshold = l.sorted(by: >)[topK - 1]
            for i in 0..<l.count where l[i] < threshold { l[i] = -.greatestFiniteMagnitude }
        }

        MathOps.softmax(&l)
        var r = Float.random(in: 0..<1)
        for (i, p) in l.enumerated() {
            r -= p
            if r < 0 { return i }
        }
        return l.count - 1
    }
}

// MARK: - Streaming generation

final class Generator: @unchecked Sendable {
    private let model: TinyGPT
    private let tokenizer: CharTokenizer

    init(model: TinyGPT, tokenizer: CharTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Streams generated text, one decoded token at a time.
    func generate(prompt: String, maxNewTokens: Int, sampler: Sampler) -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [model, tokenizer] in
                model.reset()
                var ids = tokenizer.encode(prompt)
                if ids.isEmpty { ids = [0] }

                // prefill: feed the prompt, keep only the final logits
                var logits: [Float] = []
                for id in ids {
                    if Task.isCancelled { continuation.finish(); return }
                    logits = model.step(id)
                }

                for _ in 0..<maxNewTokens {
                    if Task.isCancelled { break }
                    let next = sampler.sample(logits)
                    continuation.yield(tokenizer.decode([next]))
                    logits = model.step(next)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
