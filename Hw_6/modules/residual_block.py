import tensorflow as tf

from helpers.conv_2d import Conv2D
from helpers.group_norm import GroupNorm


class ResidualBlock(tf.Module):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel_size,
        group_norm_num_groups,
        resblock_size=2,
    ):
        """Initializes the ResidualBlock class

        Args:
            input_depth (int): the number of input channels
            output_depth (int): the number of output channels
            kernel_size (list): the kernel size
            group_norm_num_groups (int): the number of groups to split the channels into
            resblock_size (int, optional): the number of residual blocks to stack
            inbetween skip connections. Defaults to 2.
        """
        self.resblock_size = resblock_size

        self.shortcut_conv = Conv2D(input_depth, output_depth, [1, 1])

        self.conv_layers = []
        for _ in range(self.resblock_size):
            self.conv_layers.append(
                Conv2D(input_depth, output_depth, kernel_size)
            )
            input_depth = output_depth

        self.group_norm = GroupNorm(group_norm_num_groups, output_depth)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        shortcut = self.shortcut_conv(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.group_norm(x)
            x = tf.nn.relu(x)
        return x + shortcut
