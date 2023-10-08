import pickle
from pathlib import Path
import tensorflow as tf


def load_pickle_data(filename: Path, label_id: str = "labels", data_id: str = "data"):
    """Load data from a pickle file. Convert into a tensorflow dataset.
    Return the labels and images as a tuple.

    Args:
        filename (Path): Path to the pickle file.

    Returns:
        tuple: Tuple of labels and images.
    """
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    label_id = tf.cast(data[label_id.encode('utf-8')], tf.int32)
    data = tf.cast(data[data_id.encode('utf-8')], tf.float32)

    # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores
    # a 32x32 colour image. The first 1024 entries contain the red channel
    # values, the next 1024 the green, and the final 1024 the blue. The image
    # is stored in row-major order, so that the first 32 entries of the array
    # are the red channel values of the first row of the image.

    data = tf.reshape(data, [-1, 3, 32, 32])
    # convert from (batch_size, depth, height, width) to
    # (batch_size, height, width, depth)
    data = tf.transpose(data, [0, 2, 3, 1])

    data = tf.cast(data, tf.float32)/255.0
    return label_id, data
