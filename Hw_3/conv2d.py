import tensorflow as tf


class Conv2D(tf.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_shape: int,):
        """Initializes the Conv2D class

        Args:
            input_channels (int): How many channels the input has,
                e.g. for the first layer this is 3 for RGB images,
                1 for grayscale images
            output_channels (int): How many filters the convolution should have
            kernel_shape (int): Uses a filter of size
             kernel_shape x kernel_shape
        """
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2. / (input_channels + output_channels))

        self.kernel = tf.Variable(
            rng.normal(
                shape=[
                    kernel_shape[0],
                    kernel_shape[1],
                    input_channels,
                    output_channels
                ],
                stddev=stddev,
            ),
            trainable=True,
            name="Conv2D/kernel")

        self.bias = tf.Variable(
            tf.constant(
                0.01,
                shape=[output_channels]),
            trainable=True,
            name="Conv2D/bias")

    def __call__(self, input_tensor: tf.Tensor):
        """Applies the convolution to the input

        Args:
            input_tensor (tf.Tensor): The input to apply the convolution to

        Returns:
            tf.Tensor: The result of the convolution with the bias added
                the shape of the output tensor is the same as the input
        """
        return self.bias + tf.nn.conv2d(input_tensor, self.kernel, 1, "SAME")
