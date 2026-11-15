"""Import all 151 enemy front sprites from pret/pokered (task 5.3). Saves assets/species.npz.

Usage: python tools/import_species.py   (downloads gfx/pokemon/front/*.png from GitHub, once)
The PNGs are 4-grey images (white..black = shade 0..3), 40-56 px square. Names are matched to the
internal species IDs of policy/gamedata.py. The sprites are the game's own art, the same pixels the
battle screen shows (checked against live battles by evaluation/eval_battle.py).
"""
import io
import re
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixeltrainer.policy.gamedata import SPECIES  # noqa: E402

URL = "https://raw.githubusercontent.com/pret/pokered/master/gfx/pokemon/front/{}.png"
CACHE = Path("data/pret_front")


def file_name(name):
    return re.sub(r"[^a-z.]", "", name.lower().replace("♀", "f").replace("♂", "m"))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ids, names, sprites = [], [], []
    for sid, (name, *_rest) in sorted(SPECIES.items(), key=lambda kv: kv[1][1]):  # dex order
        path = CACHE / f"{file_name(name)}.png"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(URL.format(file_name(name))).read())
        g = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        s = np.zeros((56, 56), np.uint8)
        h, w = g.shape
        s[56 - h:, (56 - w + 1) // 2:(56 - w + 1) // 2 + w] = np.rint((255 - g.astype(float)) / 85)  # bottom, centred
        ids.append(sid), names.append(name), sprites.append(s)
    np.savez_compressed("assets/species.npz", ids=np.array(ids), names=np.array(names), sprites=np.stack(sprites))
    print(f"{len(ids)} sprites -> assets/species.npz")


if __name__ == "__main__":
    main()
