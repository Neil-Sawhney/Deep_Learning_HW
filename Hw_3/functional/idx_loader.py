import idx2numpy
import tensorflow as tf


def load_idx_data(filename: str,):
    """Uses idx2numpy to load an idx file into a tensor

    Args:
        filename (str): The path to the idx file

    Returns:
        tf.tensor: The tensor containing the data from the idx file
    """
    idx2numpy.convert_from_file(filename)
    return tf.convert_to_tensor(idx2numpy.convert_from_file(filename))
