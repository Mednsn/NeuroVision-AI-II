from config import (
    DATASET_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGE_SIZE
)


def test_dataset_path():

    assert DATASET_DIR.exists()


def test_number_of_classes():

    assert NUM_CLASSES == 4


def test_class_names():

    assert len(CLASS_NAMES) == 4


def test_image_size():

    assert IMAGE_SIZE == (224, 224)