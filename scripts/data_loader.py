"""
Data loading and preprocessing pipeline for aircraft damage imagery.
Handles directory generation and data augmentation using Keras ImageDataGenerator.
"""

from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_datagenerators(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42
):
    """
    Initializes ImageDataGenerators and returns training, validation, and testing generators.

    Args:
        train_dir: Path to directory containing training images.
        valid_dir: Path to directory containing validation images.
        test_dir: Path to directory containing test images.
        target_size: Tuple of (img_rows, img_cols).
        batch_size: Number of images per batch.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_generator, valid_generator, test_generator).
    """
    # Preprocessing with ResNet/VGG standard normalization (rescaling pixel values to [0, 1])
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        class_mode='binary',
        seed=seed,
        batch_size=batch_size,
        shuffle=True,
        target_size=target_size
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=valid_dir,
        class_mode='binary',
        seed=seed,
        batch_size=batch_size,
        shuffle=False,
        target_size=target_size
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        class_mode='binary',
        seed=seed,
        batch_size=batch_size,
        shuffle=False,
        target_size=target_size
    )

    return train_generator, valid_generator, test_generator
