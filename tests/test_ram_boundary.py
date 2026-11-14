"""The agent must decide from pixels only.

Nothing under pixeltrainer/ may import the evaluation package (which reads RAM),
or touch PyBoy APIs that expose internal game state instead of the screen.
"""
import ast
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent / "pixeltrainer"

FORBIDDEN_IMPORTS = {"evaluation"}
# PyBoy attributes that leak internal state. The agent may only use screen, button/tick and save/load state.
FORBIDDEN_ATTRS = {
    "memory", "memory_scanner", "symbol_lookup", "rom_symbols", "rom_symbols_inverse",
    "game_area", "game_area_collision", "game_area_mapping", "game_wrapper",
    "get_sprite", "get_sprite_by_tile_identifier", "get_tile",
    "tilemap_background", "tilemap_window", "hook_register", "gameshark",
}


def violations():
    for path in AGENT_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    yield f"{path.name}:{node.lineno} imports {name}"
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                yield f"{path.name}:{node.lineno} uses .{node.attr}"


def test_agent_reads_pixels_only():
    found = list(violations())
    assert not found, "agent code reads game state:\n" + "\n".join(found)


def test_checker_catches_violations(tmp_path, monkeypatch):
    bad = tmp_path / "bad.py"
    bad.write_text("from evaluation import truth\nx = emu.pyboy.memory[0xD35E]\n", encoding="utf-8")
    monkeypatch.setattr(__import__(__name__), "AGENT_DIR", tmp_path)
    assert len(list(violations())) == 2
