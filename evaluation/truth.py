"""RAM ground truth for evaluation and offline labelling. The agent must never import this.

Addresses come from pret/pokered's pokered.sym (load it with PyBoy(..., symbols="roms/pokered.sym")).
Species bytes are Gen 1 *internal* indices, not Pokédex numbers.
"""

SPRITE_FACING = {0x0: "down", 0x4: "up", 0x8: "left", 0xC: "right"}


def _addr(pyboy, name):
    return pyboy.symbol_lookup(name)[1]


def _u8(pyboy, name, offset=0):
    return pyboy.memory[_addr(pyboy, name) + offset]


def _u16be(pyboy, name):
    a = _addr(pyboy, name)
    return pyboy.memory[a] << 8 | pyboy.memory[a + 1]


def sprites(pyboy):
    """16 sprite slots; slot 0 is the player. Picture ID 0 means the slot is unused."""
    base1, base2 = _addr(pyboy, "wSpriteStateData1"), _addr(pyboy, "wSpriteStateData2")
    out = []
    for i in range(16):
        s1, s2 = base1 + 16 * i, base2 + 16 * i
        pic = pyboy.memory[s1]
        if pic == 0:
            continue
        sprite = {
            "slot": i,
            "picture_id": pic,
            "image_index": pyboy.memory[s1 + 2],  # 0xFF = not on screen
            "screen_y": pyboy.memory[s1 + 4],  # sprite top; the player is drawn 4 px above its cell
            "screen_x": pyboy.memory[s1 + 6],
            "facing": SPRITE_FACING.get(pyboy.memory[s1 + 9], pyboy.memory[s1 + 9]),
        }
        if i > 0:  # the game does not keep map coords for the player's slot (use wXCoord/wYCoord)
            sprite["map_y"] = pyboy.memory[s2 + 4] - 4  # stored with a +4 border offset
            sprite["map_x"] = pyboy.memory[s2 + 5] - 4
        out.append(sprite)
    return out


def read(pyboy):
    in_battle = _u8(pyboy, "wIsInBattle")  # 0 none, 1 wild, 2 trainer, 0xFF lost
    t = {
        "map_id": _u8(pyboy, "wCurMap"),
        "x": _u8(pyboy, "wXCoord"),
        "y": _u8(pyboy, "wYCoord"),
        "tileset": _u8(pyboy, "wCurMapTileset"),
        "facing": SPRITE_FACING.get(_u8(pyboy, "wSpritePlayerStateData1FacingDirection"), "?"),
        "walk_counter": _u8(pyboy, "wWalkCounter"),  # non-zero while a step is in progress
        "in_battle": in_battle,
        "battle_type": _u8(pyboy, "wBattleType"),
        "font_loaded": _u8(pyboy, "wFontLoaded"),
        "text_box_id": _u8(pyboy, "wTextBoxID"),
        "joy_ignore": _u8(pyboy, "wJoyIgnore"),
        "menu_item": _u8(pyboy, "wCurrentMenuItem"),
        "max_menu_item": _u8(pyboy, "wMaxMenuItem"),
        "status_flags5": _u8(pyboy, "wStatusFlags5"),
        "party_count": _u8(pyboy, "wPartyCount"),
        "party1": {
            "species": _u8(pyboy, "wPartyMon1Species"),
            "level": _u8(pyboy, "wPartyMon1Level"),
            "hp": _u16be(pyboy, "wPartyMon1HP"),
            "max_hp": _u16be(pyboy, "wPartyMon1MaxHP"),
        },
        "sprites": sprites(pyboy),
    }
    if in_battle:
        t["enemy"] = {
            "species": _u8(pyboy, "wEnemyMonSpecies"),
            "level": _u8(pyboy, "wEnemyMonLevel"),
            "hp": _u16be(pyboy, "wEnemyMonHP"),
            "max_hp": _u16be(pyboy, "wEnemyMonMaxHP"),
        }
        t["own"] = {
            "species": _u8(pyboy, "wBattleMonSpecies"),
            "level": _u8(pyboy, "wBattleMonLevel"),
            "hp": _u16be(pyboy, "wBattleMonHP"),
            "max_hp": _u16be(pyboy, "wBattleMonMaxHP"),
            "moves": [_u8(pyboy, "wBattleMonMoves", i) for i in range(4)],
            "pp": [_u8(pyboy, "wBattleMonPP", i) & 0x3F for i in range(4)],  # top 2 bits are PP Ups
        }
    return t


def tilemap(pyboy):
    """The 20x18 screen tile buffer (wTileMap) as a list of rows. Text tiles here label OCR truth."""
    a = _addr(pyboy, "wTileMap")
    return [[pyboy.memory[a + 20 * r + c] for c in range(20)] for r in range(18)]
