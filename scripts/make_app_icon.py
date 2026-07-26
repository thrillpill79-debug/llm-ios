"""Generate the 1024x1024 app icon used by both iOS apps.

Written with zlib/struct so it needs no image libraries, and emits RGB with no
alpha channel — the App Store rejects icons that have one.

    python3 scripts/make_app_icon.py
"""
import os
import struct
import zlib

W = H = 1024

# same pixel-art "spark" as the web app's icon
GLYPH = [
    "....11....",
    "....11....",
    ".1..11..1.",
    "..1.11.1..",
    "...1111...",
    "1111111111",
    "1111111111",
    "...1111...",
    "..1.11.1..",
    ".1..11..1.",
    "....11....",
    "....11....",
]
GW, GH = len(GLYPH[0]), len(GLYPH)
CELL = 62
OX = (W - GW * CELL) // 2
OY = (H - GH * CELL) // 2


def background(y):
    """Vertical gradient #16162a -> #232345."""
    t = y / H
    return (int(0x16 + (0x23 - 0x16) * t),
            int(0x16 + (0x23 - 0x16) * t),
            int(0x2A + (0x45 - 0x2A) * t))


def build_png() -> bytes:
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter type: none
        bg = bytes(background(y))
        gy = (y - OY) // CELL
        row_lit = GLYPH[gy] if 0 <= gy < GH else None
        for x in range(W):
            gx = (x - OX) // CELL
            if row_lit is not None and 0 <= gx < GW and row_lit[gx] == "1":
                raw += b"\x9c\x8c\xff"
            else:
                raw += bg

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def main():
    png = build_png()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for app in ("LlamaChat", "TinyLLM"):
        out = os.path.join(root, "ios", app, "Assets.xcassets", "AppIcon.appiconset")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "icon-1024.png"), "wb") as f:
            f.write(png)
    print(f"wrote {len(png)} byte icon for LlamaChat and TinyLLM")


if __name__ == "__main__":
    main()
