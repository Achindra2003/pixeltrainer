"""Check the ROM: hash, header title, and that it boots in PyBoy.

Usage: python tools/check_rom.py [roms/pokered.gb]
Writes runs/boot.png (frame after ~10 s of emulated time) for a visual check.
"""
import hashlib
import sys
from pathlib import Path

import cv2
from pyboy import PyBoy

CLEAN_RED_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"  # Pokémon Red (UE), unmodified dump


def main(rom: Path) -> int:
    data = rom.read_bytes()
    sha1 = hashlib.sha1(data).hexdigest()
    title = data[0x134:0x144].split(b"\0")[0].decode("ascii", "replace")
    print(f"file   {rom} ({len(data)} bytes)")
    print(f"title  {title!r}")
    print(f"sha1   {sha1}  {'(clean Red)' if sha1 == CLEAN_RED_SHA1 else '(NOT the clean Red dump)'}")
    if title != "POKEMON RED":
        print("error: header title is not POKEMON RED")
        return 1

    pyboy = PyBoy(str(rom), window="null")
    pyboy.set_emulation_speed(0)
    pyboy.tick(600, True)  # ~10 s of emulated time
    frame = pyboy.screen.ndarray  # (144, 160, 4) RGBA
    pyboy.stop(save=False)

    out = Path("runs/boot.png")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
    print(f"boot   ok, frame {frame.shape} saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "roms/pokered.gb")))
