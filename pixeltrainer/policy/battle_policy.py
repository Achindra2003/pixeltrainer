"""Move choice (PLAN.md §4.8, task 5.6). Each policy returns (move index, reason text for the live view).

Inputs are what perception read: the move names in the fight menu, the own species (from the name on
screen) and the enemy species (sprite match). Expected damage = power x STAB x effectiveness x accuracy.
Status moves (power 0) are worth STATUS_VALUE, so they win only when nothing deals damage.
"""
import numpy as np

from pixeltrainer.policy.gamedata import MOVE_ID, MOVES, SPECIES, SPECIES_ID, effectiveness

STATUS_VALUE = 1.0
POLICIES = ("random", "greedy", "type")


def _move(name):
    return MOVES[MOVE_ID[name]]  # (name, type, power, accuracy, pp)


def expected(move, own_types, enemy_types):
    _, mtype, power, acc, _ = _move(move)
    # shortcut: fixed-damage moves (power 1: SONICBOOM, SEISMIC TOSS, ...) count as status; route 1-2 starters lack them.
    if power <= 1:
        return STATUS_VALUE, 1.0
    eff = effectiveness(mtype, enemy_types)
    return power * (1.5 if mtype in own_types else 1.0) * eff * acc / 100, eff


def choose(policy, moves, own="", enemy="", rng=None):
    """moves: names on screen; own/enemy: species names. Returns (index into moves, reason)."""
    if policy == "random":
        k = int((rng or np.random.default_rng()).integers(len(moves)))
        return k, f"{moves[k]}: random"
    if policy == "greedy":  # highest base power, types ignored
        k = max(range(len(moves)), key=lambda i: _move(moves[i])[2])
        return k, f"{moves[k]}: power {_move(moves[k])[2]}"
    own_types = SPECIES[SPECIES_ID[own]][2:] if own in SPECIES_ID else ()
    enemy_types = SPECIES[SPECIES_ID[enemy]][2:] if enemy in SPECIES_ID else ()
    scored = [expected(m, own_types, enemy_types) for m in moves]
    k = max(range(len(moves)), key=lambda i: scored[i][0])
    return k, f"{moves[k]} vs {enemy.title()}: {scored[k][1]:g}x -> exp {scored[k][0]:.0f}"
