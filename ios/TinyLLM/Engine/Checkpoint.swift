// Loads the exported model (config.json, vocab.json, manifest.json, weights.bin)
// produced by python/export_ios.py.

import Foundation

struct ModelConfig: Decodable {
    let vocab_size: Int
    let block_size: Int
    let n_layer: Int
    let n_head: Int
    let n_embd: Int
}

struct Checkpoint {
    let config: ModelConfig
    let chars: [String]
    let tensors: [String: [Float]]

    enum LoadError: Error, LocalizedError {
        case missingResource(String)
        case badWeights(String)

        var errorDescription: String? {
            switch self {
            case .missingResource(let name):
                return "Missing model resource '\(name)'. Add the model/export folder to the app target."
            case .badWeights(let why):
                return "Corrupt weights.bin: \(why)"
            }
        }
    }

    private struct ManifestEntry: Decodable {
        let name: String
        let shape: [Int]
        let offset: Int
        let count: Int
    }

    private struct Vocab: Decodable { let chars: [String] }

    static func load(from bundle: Bundle = .main) throws -> Checkpoint {
        try load { name, ext in
            guard let url = bundle.url(forResource: name, withExtension: ext) else {
                throw LoadError.missingResource("\(name).\(ext)")
            }
            return try Data(contentsOf: url)
        }
    }

    /// Load from a plain directory containing the four exported files
    /// (used by the SwiftPM tests; the app uses load(from bundle:)).
    static func load(fromDirectory dir: URL) throws -> Checkpoint {
        try load { name, ext in
            let url = dir.appendingPathComponent("\(name).\(ext)")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw LoadError.missingResource(url.path)
            }
            return try Data(contentsOf: url)
        }
    }

    private static func load(_ data: (String, String) throws -> Data) throws -> Checkpoint {
        let decoder = JSONDecoder()
        let config = try decoder.decode(ModelConfig.self, from: data("config", "json"))
        let vocab = try decoder.decode(Vocab.self, from: data("vocab", "json"))
        let manifest = try decoder.decode([ManifestEntry].self, from: data("manifest", "json"))
        let raw = try data("weights", "bin")

        let totalFloats = raw.count / MemoryLayout<Float>.size
        guard let last = manifest.last, last.offset + last.count <= totalFloats else {
            throw LoadError.badWeights("file has \(totalFloats) floats, manifest wants more")
        }

        // one contiguous copy, then slice per tensor (weights are little-endian
        // float32; all Apple platforms are little-endian)
        let all = [Float](unsafeUninitializedCapacity: totalFloats) { buf, count in
            _ = raw.copyBytes(to: buf)
            count = totalFloats
        }

        var tensors: [String: [Float]] = [:]
        for entry in manifest {
            tensors[entry.name] = Array(all[entry.offset..<(entry.offset + entry.count)])
        }
        return Checkpoint(config: config, chars: vocab.chars, tensors: tensors)
    }
}
