import tensorflow as tf


class GroupNorm(tf.Module):
    def __init__(
        self,
        num_groups: int,
        input_depth: int,
        input_channels: int = 2,
        epsilon: float = 1e-5,
    ):
        """Initializes the GroupNorm class

        Args:
            num_groups (int): the number of groups to split the channels into
            input_depth (int): the depth of the input tensor
            input_channels (int): the number of channels in the input tensor
            epsilon (float, optional): small value for numerical stability.
                Defaults to 1e-5.
        """
        self.num_groups = int(num_groups)
        self.epsilon = epsilon
        self.input_depth = input_depth
        self.input_channels = input_channels

        if input_channels == 1:
            self.gamma = tf.Variable(
                tf.ones(shape=[1, 1, input_depth]),
                trainable=True,
                name="GroupNorm/gamma",
            )

            self.beta = tf.Variable(
                tf.zeros(shape=[1, 1, input_depth]),
                trainable=True,
                name="GroupNorm/beta",
            )

        elif input_channels == 2:
            self.gamma = tf.Variable(
                tf.ones(shape=[1, 1, 1, input_depth]),
                trainable=True,
                name="GroupNorm/gamma",
            )

            self.beta = tf.Variable(
                tf.zeros(shape=[1, 1, 1, input_depth]),
                trainable=True,
                name="GroupNorm/beta",
            )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Applies group normalization to the input tensor

        Args:
            x (tf.Tensor): the input tensor

        Returns:
            tf.Tensor: the normalized tensor
        """
        if self.input_channels == 1:
            (
                batch_size,
                context_length,
                model_dim,
            ) = x.shape
            x = tf.reshape(
                x,
                [
                    batch_size,
                    context_length,
                    self.num_groups,
                    model_dim // self.num_groups,
                ],
            )

            mean, variance = tf.nn.moments(x, axes=[1, 3], keepdims=True)
            x = (x - mean) / tf.sqrt(variance + self.epsilon)

            x = tf.reshape(x, [batch_size, context_length, model_dim])

        elif self.input_channels == 2:
            (
                batch_size,
                input_height,
                input_width,
                input_depth,
            ) = x.shape
            x = tf.reshape(
                x,
                [
                    batch_size,
                    input_height,
                    input_width,
                    self.num_groups,
                    input_depth // self.num_groups,
                ],
            )

            mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
            x = (x - mean) / tf.sqrt(variance + self.epsilon)

            x = tf.reshape(x, [batch_size, input_height, input_width, input_depth])

        return x * self.gamma + self.beta
