"""Character-level tokenizer.

Every distinct character in the training corpus becomes one token. Simple,
dependency-free, and easy to reimplement anywhere (see CharTokenizer.swift).
Swap this for a BPE tokenizer later if you scale up the model.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # unknown characters are dropped rather than crashing generation
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            return cls(json.load(f)["chars"])
