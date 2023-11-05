import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.Module):
    """PositionalEncoding layer.

    This is an implementation of positional encoding as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017).

    This layer first calculates a positional encoding matrix, then adds it to
    the inputs.

    Args:
        max_position: Maximum position to encode.
        model_dim: Size of each attention head for value, query, and queue.

    Call arguments:
        inputs: Input `Tensor` of shape `(B, seq_len, model_dim)`.

    Returns:
        output: The result of the computation, of shape `(B, seq_len, model_dim)`,
    """

    def __init__(self, max_position, model_dim):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self._calculate_positional_encoding(
            max_position, model_dim
        )

    def _calculate_positional_encoding(self, max_position, model_dim):
        positions = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(-np.arange(0, model_dim, 2) * (np.log(10000.0) / model_dim))
        positional_encoding = np.zeros((max_position, model_dim))
        positional_encoding[:, 0::2] = np.sin(positions * div_term)
        positional_encoding[:, 1::2] = np.cos(positions * div_term)
        return tf.convert_to_tensor(
            positional_encoding[np.newaxis, ...], dtype=tf.float32
        )

    def __call__(self, inputs):
        return inputs + self.positional_encoding[:, : inputs.shape[1], :]
