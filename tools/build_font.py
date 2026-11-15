"""Font templates from the ROM (task 5.1). Saves assets/font.npz: chars and 8x8 shade glyphs.

Usage: python tools/build_font.py
FontGraphics (pret/pokered symbol) is 1 bpp, tile IDs 0x80-0xFF; bit 1 = black (shade 3). The text glyphs
match the 8x8 tiles captured from our own recordings (assets/tile_glyphs.npz, 74 of 74), so these are the
same pixels the agent sees. Character names follow pret's charmap.asm.
"""
import numpy as np

CHARS = {0x7F: " ", 0x9A: "(", 0x9B: ")", 0x9C: ":", 0x9D: ";", 0x9E: "[", 0x9F: "]", 0xBA: "é",
         0xE0: "'", 0xE1: "<PK>", 0xE2: "<MN>", 0xE3: "-", 0xE6: "?", 0xE7: "!", 0xE8: ".", 0xEC: "▷",
         0xED: "▶", 0xEE: "▼", 0xEF: "♂", 0xF0: "¥", 0xF1: "×", 0xF3: "/", 0xF4: ",", 0xF5: "♀",
         **{0x80 + i: chr(65 + i) for i in range(26)}, **{0xA0 + i: chr(97 + i) for i in range(26)},
         **{0xF6 + i: str(i) for i in range(10)}}


def main():
    sym = dict(reversed(line.split(" ", 1)) for line in open("roms/pokered.sym") if line[:1] != ";" and " " in line)
    bank, addr = (int(v, 16) for v in sym["FontGraphics\n"].split(":"))
    off = bank * 0x4000 + addr - 0x4000
    rom = open("roms/pokered.gb", "rb").read()
    bits = np.unpackbits(np.frombuffer(rom[off:off + 0x400], np.uint8).reshape(128, 8), axis=1)
    glyphs = (bits.reshape(128, 8, 8) * 3).astype(np.uint8)
    ids = sorted(CHARS)
    np.savez_compressed("assets/font.npz", chars=np.array([CHARS[i] for i in ids]), ids=np.array(ids),
                        glyphs=np.stack([np.zeros((8, 8), np.uint8) if i == 0x7F else glyphs[i - 0x80] for i in ids]))
    print(f"{len(ids)} glyphs -> assets/font.npz")


if __name__ == "__main__":
    main()
