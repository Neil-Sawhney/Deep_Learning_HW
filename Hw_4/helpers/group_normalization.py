import tensorflow as tf


# TODO: check if this is right
def group_normalization(x: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor,
                        G: int = 32, eps: int = 1e-5):
    """Group Normalization

    Args:
        x (tf.Tensor): The input tensor
        gamma (tf.float32): The gamma parameter
        beta (tf.float32): The beta parameter
        G (int, optional): The number of groups. Defaults to 32.
        eps (float, optional): The epsilon parameter. Defaults to 1e-5.

    Returns:
        tf.Tensor: The normalized tensor
    """
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
