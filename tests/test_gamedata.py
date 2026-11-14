import pytest

from pixeltrainer.policy import gamedata as g
from tools import gen_gamedata as rom


def test_gen1_matchups():
    assert g.effectiveness("GHOST", ["PSYCHIC"]) == 0  # Gen 1 bug
    assert g.effectiveness("BUG", ["POISON"]) == 2
    assert g.effectiveness("POISON", ["BUG"]) == 2
    assert g.effectiveness("ICE", ["FIRE"]) == 1
    assert g.effectiveness("FIRE", ["GRASS", "POISON"]) == 2  # Bulbasaur
    assert g.effectiveness("ELECTRIC", ["WATER", "FLYING"]) == 4  # Gyarados
    assert g.effectiveness("NORMAL", ["ROCK", "GROUND"]) == 0.5  # single entry per distinct type
    assert g.effectiveness("FIRE", ["FIRE", "FIRE"]) == 0.5  # single-type species listed twice


def test_known_entries():
    assert g.SPECIES[176][:3] == ("CHARMANDER", 4, "FIRE")
    assert g.SPECIES_ID["PIDGEY"] == 36
    assert g.MOVES[10] == ("SCRATCH", "NORMAL", 40, 100, 35)
    assert g.MOVE_ID["EMBER"] == 52 and g.MOVES[52][1:3] == ("FIRE", 40)
    assert len(g.SPECIES) == 151 and len(g.MOVES) == 165


@pytest.mark.skipif(not rom.ROM, reason="needs roms/pokered.gb")
def test_tables_match_rom():
    names = rom.type_names()
    assert g.CHART == {(names[a], names[d]): m for (a, d), m in rom.type_chart().items()}
    assert g.MOVES == {i: (n, names[t], p, a, pp) for i, (n, t, p, a, pp) in rom.moves().items()}
    assert g.SPECIES == {i: (n, dex, names[t1], names[t2]) for i, (n, dex, t1, t2) in rom.species().items()}
