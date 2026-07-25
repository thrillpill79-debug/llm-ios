// Numerical parity tests: the Swift engine must reproduce the PyTorch/NumPy
// reference implementations on the committed demo model (model/export/).
// Reference values were computed with python/verify_export.py's NumpyGPT,
// which itself is verified against the PyTorch checkpoint.

import XCTest
@testable import TinyLLMEngine

final class TinyGPTParityTests: XCTestCase {

    // "ROMEO:" under the committed Shakespeare vocab
    static let promptIDs = [30, 27, 25, 17, 27, 10]

    // logits after feeding the prompt, from the NumPy reference
    static let expectedLogits: [Float] = [
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
    ]

    // 20 greedy tokens after the prompt ("\nI would the shall t");
    // minimum argmax margin along this path is 0.027, far above fp32 noise
    static let expectedGreedy = [0, 21, 1, 61, 53, 59, 50, 42, 1, 58,
                                 46, 43, 1, 57, 46, 39, 50, 50, 1, 58]

    private func loadCheckpoint() throws -> Checkpoint {
        let env = ProcessInfo.processInfo.environment["LLM_EXPORT_DIR"]
        let dir = URL(fileURLWithPath: env ?? "model/export")
        return try Checkpoint.load(fromDirectory: dir)
    }

    func testTokenizerRoundTrip() throws {
        let checkpoint = try loadCheckpoint()
        let tokenizer = CharTokenizer(chars: checkpoint.chars)
        XCTAssertEqual(tokenizer.encode("ROMEO:"), Self.promptIDs)
        XCTAssertEqual(tokenizer.decode(Self.promptIDs), "ROMEO:")
    }

    func testLogitsMatchReference() throws {
        let model = TinyGPT(checkpoint: try loadCheckpoint())
        var logits: [Float] = []
        for id in Self.promptIDs {
            logits = model.step(id)
        }
        XCTAssertEqual(logits.count, Self.expectedLogits.count)
        var worst: Float = 0
        for (got, want) in zip(logits, Self.expectedLogits) {
            worst = max(worst, abs(got - want))
        }
        XCTAssertLessThan(worst, 5e-3, "logits diverge from NumPy reference")
    }

    func testGreedyGenerationMatchesReference() throws {
        let model = TinyGPT(checkpoint: try loadCheckpoint())
        var logits: [Float] = []
        for id in Self.promptIDs {
            logits = model.step(id)
        }
        var produced: [Int] = []
        for _ in 0..<Self.expectedGreedy.count {
            var best = 0
            for i in 1..<logits.count where logits[i] > logits[best] { best = i }
            produced.append(best)
            logits = model.step(best)
        }
        XCTAssertEqual(produced, Self.expectedGreedy)
    }

    func testContextOverflowReprefill() throws {
        let checkpoint = try loadCheckpoint()
        let model = TinyGPT(checkpoint: checkpoint)
        // run well past block_size; must not crash and must keep producing
        // finite logits through the re-prefill path
        var logits = model.step(0)
        for i in 0..<(checkpoint.config.block_size + 40) {
            logits = model.step(i % checkpoint.config.vocab_size)
        }
        XCTAssertTrue(logits.allSatisfy { $0.isFinite })
        XCTAssertLessThanOrEqual(model.cachedTokens.count, checkpoint.config.block_size)
    }

    func testUTF8StreamDecoder() {
        // "é" (0xC3 0xA9) split across two pieces: the complete "a" is
        // emitted immediately, the dangling lead byte is held back
        var decoder = UTF8PieceDecoder()
        XCTAssertEqual(decoder.feed([0x61, 0xC3]), "a")
        XCTAssertEqual(decoder.feed([0xA9]), "é")
        // 4-byte emoji split 1+3
        decoder = UTF8PieceDecoder()
        XCTAssertNil(decoder.feed([0xF0]))
        XCTAssertEqual(decoder.feed([0x9F, 0x98, 0x80]), "😀")
        // plain ASCII passes straight through
        decoder = UTF8PieceDecoder()
        XCTAssertEqual(decoder.feed(Array("hello".utf8)), "hello")
    }
}
