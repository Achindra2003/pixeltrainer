import numpy as np
import pytest

from pixeltrainer.perception import tiles


def test_passable_matches_label_classes():
    assert tiles.PASSABLE <= set(tiles.CLASSES)


@pytest.mark.skipif(not tiles.MODEL.exists(), reason="needs models/tiles.pt (tools/train_tiles.py)")
def test_cnn_labels_every_cell():
    labels, conf = tiles.cnn(np.zeros((144, 160), np.uint8))
    assert labels.shape == conf.shape == (9, 10)
    assert set(labels.ravel()) <= set(tiles.CLASSES) and ((conf > 0) & (conf <= 1)).all()
