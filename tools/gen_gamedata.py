"""Generate pixeltrainer/policy/gamedata.py from the ROM (tables located via roms/pokered.sym).

Usage: python tools/gen_gamedata.py
The ROM readers are also used by tests/test_gamedata.py to check the generated tables.
"""
from pathlib import Path

ROM = Path("roms/pokered.gb").read_bytes() if Path("roms/pokered.gb").exists() else b""
SYM = {}
for line in (Path("roms/pokered.sym").read_text().splitlines() if ROM else []):
    if line and line[0] != ";" and ":" in line.split(" ")[0]:
        loc, name = line.split(" ", 1)
        bank, addr = loc.split(":")
        SYM[name] = int(bank, 16) * 0x4000 + int(addr, 16) - (0x4000 if int(bank, 16) else 0)

CHARS = {0x7F: " ", 0xE0: "'", 0xE3: "-", 0xE6: "?", 0xE7: "!", 0xE8: ".", 0xEF: "♂", 0xF3: "/", 0xF4: ",", 0xF5: "♀",
         **{0x80 + i: chr(65 + i) for i in range(26)}, **{0xA0 + i: chr(97 + i) for i in range(26)},
         **{0xF6 + i: str(i) for i in range(10)}}


def text(off):
    out = ""
    while ROM[off] != 0x50:
        out += CHARS.get(ROM[off], f"<{ROM[off]:02X}>")
        off += 1
    return out, off + 1


def type_names():
    base, bank = SYM["TypeNames"], SYM["TypeNames"] // 0x4000
    out = {}
    for t in range(0x1B):
        ptr = ROM[base + 2 * t] | ROM[base + 2 * t + 1] << 8
        name, _ = text(bank * 0x4000 + ptr - 0x4000)
        out[t] = name
    return out


def type_chart():
    off, out = SYM["TypeEffects"], {}
    while ROM[off] != 0xFF:
        a, d, m = ROM[off:off + 3]
        out[(a, d)] = m / 10
        off += 3
    return out


def moves():
    off = SYM["MoveNames"]
    names = []
    for _ in range(165):
        n, off = text(off)
        names.append(n)
    out = {}
    for i, n in enumerate(names):
        anim, _effect, power, typ, acc, pp = ROM[SYM["Moves"] + 6 * i:SYM["Moves"] + 6 * i + 6]
        assert anim == i + 1
        out[i + 1] = (n, typ, power, round(acc * 100 / 255), pp)
    return out


def species():
    out = {}
    for internal in range(1, 191):
        dex = ROM[SYM["PokedexOrder"] + internal - 1]
        if not dex:
            continue
        name = bytes(ROM[SYM["MonsterNames"] + 10 * (internal - 1):][:10])
        name = "".join(CHARS.get(b, "") for b in name.split(b"\x50")[0])
        base = SYM["MewBaseStats"] if dex == 151 else SYM["BaseStats"] + 28 * (dex - 1)
        assert ROM[base] == dex
        out[internal] = (name, dex, ROM[base + 6], ROM[base + 7])
    return out




def main():
    tn = type_names()
    chart = type_chart()
    L = ['"""Gen 1 static game tables (type chart, moves, species), generated once from the ROM with pret/pokered',
         'symbols and checked against it by tests/test_gamedata.py. Static data only: nothing here reads game state.',
         '"""', "",
         "# Type effectiveness in table order; pairs not listed are 1x. Gen 1 quirks kept: Ghost vs Psychic = 0,",
         "# Bug vs Poison = 2, Poison vs Bug = 2, Ice vs Fire = 1.",
         "CHART = {"]
    for (a, d), m in chart.items():
        L.append(f'    ("{tn[a]}", "{tn[d]}"): {m},')
    L += ["}", "", "# Move ID -> (name, type, power, accuracy %, PP). Power 0 = status move; 1 = fixed-damage move.", "MOVES = {"]
    for i, (n, t, p, a, pp) in moves().items():
        L.append(f'    {i}: ("{n}", "{tn[t]}", {p}, {a}, {pp}),')
    L += ["}", "MOVE_ID = {v[0]: k for k, v in MOVES.items()}", "",
          "# Species internal index (as RAM and battle code use it) -> (name, Pokedex number, type 1, type 2).",
          "SPECIES = {"]
    for i, (n, dex, t1, t2) in species().items():
        L.append(f'    {i}: ("{n}", {dex}, "{tn[t1]}", "{tn[t2]}"),')
    L += ["}", "SPECIES_ID = {v[0]: k for k, v in SPECIES.items()}", "", "",
          "def effectiveness(move_type, defender_types):",
          '    """Damage multiplier of a move type against a defender (types of a single-type species are listed twice)."""',
          "    m = 1.0",
          "    for t in set(defender_types):",
          "        m *= CHART.get((move_type, t), 1.0)",
          "    return m", ""]
    open("pixeltrainer/policy/gamedata.py", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
