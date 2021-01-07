import tensorflow as tf
import numpy as np
import random

from tensorflow.keras import layers


def set_seed(seed=100):
    """
    Set global seed

    Args:
        seed: seed number
    """

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initial(seed=100):
    """
    Initial seed & gpu

    Args:
        seed: seed number
    """

    set_seed(seed)

    if tf.test.is_gpu_available():  # gpu limit
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def make_mlp(sizes, activation, output_activation=None):
    """
    Create MLP

    Args:
        sizes: unit size for each layer
        activation: activation for apply each layer except last layer
        output_activation: activation for last layer

    Returns:
        layer block
    """

    l = [layers.Input(sizes[0])]
    for i in range(1, len(sizes)):
        if i != len(sizes) - 1:
            l.append(layers.Dense(sizes[i], activation=activation))
        else:
            l.append(layers.Dense(sizes[i], activation=output_activation))
    return tf.keras.Sequential(l)
