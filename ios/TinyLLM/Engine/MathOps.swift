// Small float math kernels for on-device inference.
// Uses Accelerate (vDSP) when available, with portable pure-Swift fallbacks.

import Foundation
#if canImport(Accelerate)
import Accelerate
#endif
#if canImport(Glibc)
import Glibc   // erff/exp/sqrt on Linux (CI test builds)
#endif

enum MathOps {

    /// y = W·x + b where W is [rows x cols] row-major (PyTorch Linear layout).
    static func matVec(_ w: [Float], rows: Int, cols: Int, _ x: [Float], bias: [Float]? = nil) -> [Float] {
        precondition(x.count == cols && w.count == rows * cols)
        var y = [Float](repeating: 0, count: rows)
        #if canImport(Accelerate)
        w.withUnsafeBufferPointer { wp in
            x.withUnsafeBufferPointer { xp in
                y.withUnsafeMutableBufferPointer { yp in
                    vDSP_mmul(wp.baseAddress!, 1, xp.baseAddress!, 1, yp.baseAddress!, 1,
                              vDSP_Length(rows), 1, vDSP_Length(cols))
                }
            }
        }
        #else
        for r in 0..<rows {
            var acc: Float = 0
            let base = r * cols
            for c in 0..<cols { acc += w[base + c] * x[c] }
            y[r] = acc
        }
        #endif
        if let bias {
            for r in 0..<rows { y[r] += bias[r] }
        }
        return y
    }

    static func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
        #if canImport(Accelerate)
        var out: Float = 0
        vDSP_dotpr(a, 1, b, 1, &out, vDSP_Length(n))
        return out
        #else
        var acc: Float = 0
        for i in 0..<n { acc += a[i] * b[i] }
        return acc
        #endif
    }

    /// In-place LayerNorm matching torch.nn.LayerNorm (population variance, eps 1e-5).
    static func layerNorm(_ x: inout [Float], gain: [Float], bias: [Float], eps: Float = 1e-5) {
        let n = Float(x.count)
        var mean: Float = 0
        for v in x { mean += v }
        mean /= n
        var variance: Float = 0
        for v in x { let d = v - mean; variance += d * d }
        variance /= n
        let inv = 1.0 / sqrt(variance + eps)
        for i in 0..<x.count {
            x[i] = (x[i] - mean) * inv * gain[i] + bias[i]
        }
    }

    /// In-place softmax over the whole vector.
    static func softmax(_ x: inout [Float]) {
        var maxv = -Float.greatestFiniteMagnitude
        for v in x where v > maxv { maxv = v }
        var sum: Float = 0
        for i in 0..<x.count {
            let e = exp(x[i] - maxv)
            x[i] = e
            sum += e
        }
        for i in 0..<x.count { x[i] /= sum }
    }

    /// Exact GELU (erf form), matching torch.nn.functional.gelu's default.
    static func gelu(_ x: inout [Float]) {
        for i in 0..<x.count {
            x[i] = 0.5 * x[i] * (1.0 + erff(x[i] / 1.4142135))
        }
    }
}
