// Character-level tokenizer mirroring python/tokenizer.py.

import Foundation

struct CharTokenizer {
    let chars: [String]
    private let stoi: [String: Int]

    init(chars: [String]) {
        self.chars = chars
        var map: [String: Int] = [:]
        for (i, ch) in chars.enumerated() { map[ch] = i }
        self.stoi = map
    }

    /// Characters the model has never seen are dropped, same as the Python side.
    func encode(_ text: String) -> [Int] {
        text.compactMap { stoi[String($0)] }
    }

    func decode(_ ids: [Int]) -> String {
        ids.map { chars[$0] }.joined()
    }
}
