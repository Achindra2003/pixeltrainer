from pixeltrainer.policy.battle_policy import choose


def test_type_aware_prefers_super_effective_over_power():
    moves = ["TACKLE", "EMBER"]  # 35 vs 40 power, but EMBER is 2x on grass and gets STAB
    assert choose("type", moves, "CHARMANDER", "BULBASAUR")[0] == 1
    assert choose("type", ["SCRATCH", "EMBER"], "CHARMANDER", "SQUIRTLE")[0] == 0  # 0.5x: SCRATCH 40 > EMBER 30


def test_status_moves_only_when_nothing_damages():
    assert choose("type", ["GROWL", "SCRATCH"], "CHARMANDER", "RATTATA")[0] == 1
    assert choose("type", ["GROWL", "TAIL WHIP"], "CHARMANDER", "RATTATA")[0] == 0


def test_ghost_immune_to_normal():
    k, reason = choose("type", ["TACKLE", "GROWL"], "SQUIRTLE", "GASTLY")
    assert k == 1 and "GROWL" in reason


def test_greedy_and_random():
    assert choose("greedy", ["GROWL", "SCRATCH"])[0] == 1
    assert choose("random", ["GROWL", "SCRATCH"])[0] in (0, 1)


def test_misread_move_name_snaps_to_closest_or_counts_as_status():
    from pixeltrainer.policy.battle_policy import _move
    assert _move("G  WL")[0] == "GROWL"  # OCR on a degraded capture (task 8.7)
    assert _move("xx")[2] == 0
