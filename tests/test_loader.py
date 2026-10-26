import numpy as np

from src.data.loader import (
    load_dataset,
    get_image_paths
)


def test_image_paths():

    files = get_image_paths()

    assert len(files) > 0


def test_load_dataset():

    X, y = load_dataset()

    assert len(X) == len(y)

    assert X.ndim == 4

    assert X.shape[1:] == (224, 224, 3)

    assert X.dtype == np.uint8