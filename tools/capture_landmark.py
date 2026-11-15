"""Cut the Pokémon Center "POKé" sign out of the Viridian start frame (task 4.3).

Usage: python tools/capture_landmark.py
Saves assets/landmarks/poke_sign.png (shade s stored as 255 - 85 s). Only the Pokémon Center has this
sign (the Mart reads "MART"), and there is none in Pallet Town or on Route 1, so seeing it = Viridian.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixeltrainer.emu import Emu  # noqa: E402

emu = Emu(state="states/viridian_north.state")
frame = emu.reset(0)
emu.close()
cv2.imwrite("assets/landmarks/poke_sign.png", 255 - 85 * frame[48:60, 78:96])
print("saved assets/landmarks/poke_sign.png")
