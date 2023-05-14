import tensorflow as tf
from typing import Sized
import ipdb


def get_data_loaders(
    *, train_batch_size=12, test_batch_size=12, save_dir
) -> tuple[Sized, Sized]:
    # Load the CIFAR10 dataset
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.cifar10.load_data()

    # Create the dataloaders
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    ).batch(train_batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
        test_batch_size
    )
    return iter(train_dataset), iter(test_dataset)
