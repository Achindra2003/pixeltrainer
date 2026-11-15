import numpy as np

from pixeltrainer.perception import ocr


def draw(text, row=14, col=1):
    frame = np.zeros((144, 160), np.uint8)
    for k, ch in enumerate(text):
        frame[8 * row:8 * row + 8, 8 * (col + k):8 * (col + k) + 8] = ocr.GLYPHS[ocr.CHARS.index(ch)].reshape(8, 8)
    return frame


def test_reads_drawn_text_and_cursor():
    f = draw("▶TACKLE  35/35")
    assert ocr.read(f, 14) == "▶TACKLE  35/35"
    assert ocr.cursor(f) == [(14, 1)]


def test_tolerates_a_few_wrong_pixels():
    f = draw("Wild PIDGEY")
    f[14 * 8 + 3, 8 * 2 + 3] ^= 3  # flip one pixel of "i"
    assert ocr.read(f, 14) == "Wild PIDGEY"


def test_non_text_is_none():
    f = np.random.default_rng(0).integers(0, 4, (144, 160)).astype(np.uint8)
    assert ocr.read_tile(ocr.tiles(f)[0, 0]) is None
