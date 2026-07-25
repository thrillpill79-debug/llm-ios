// Reassembles UTF-8 text from token pieces that may split multi-byte
// characters (CJK, emoji) across token boundaries. Used by the LlamaChat
// engine; kept dependency-free so it is unit-tested in the SwiftPM package.

import Foundation

struct UTF8PieceDecoder {
    private var pending: [UInt8] = []

    mutating func feed(_ bytes: [UInt8]) -> String? {
        pending.append(contentsOf: bytes)

        // count trailing bytes that form an incomplete UTF-8 character
        var holdback = 0
        var i = pending.count - 1
        while i >= 0 && holdback < 4 {
            let b = pending[i]
            if b & 0xC0 != 0x80 { // found the lead byte (or ASCII)
                let expected: Int
                if b & 0x80 == 0 { expected = 1 }
                else if b & 0xE0 == 0xC0 { expected = 2 }
                else if b & 0xF0 == 0xE0 { expected = 3 }
                else { expected = 4 }
                let have = pending.count - i
                holdback = have < expected ? have : 0
                break
            }
            i -= 1
            holdback += 1
        }
        if holdback >= pending.count { return nil }

        let ready = Array(pending[0..<(pending.count - holdback)])
        pending.removeFirst(ready.count)
        return ready.isEmpty ? nil : String(decoding: ready, as: UTF8.self)
    }
}
