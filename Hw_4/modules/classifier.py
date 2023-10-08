import tensorflow as tf

from helpers.conv_2d import Conv2D
from helpers.group_norm import GroupNorm

from modules.mlp import MLP


class Classifier(tf.Module):
    def __init__(
        self,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
        input_size: int,
        resblock_size: int = 2,
        pool_size: int = 2,
        dropout_prob: float = 0.5,
        group_norm_num_groups: int = 32,
        num_hidden_layers: int = 1,
        hidden_layer_width: int = 128,
    ):
        """Initializes the Classifier class

        Args:
            input_depth (int): number of input channels,
                e.g. this is 3 for RGB images, 1 for grayscale images
            layer_depths (list[int]): A list of how many filters each layer
                should have the length of this list determines how many layers
                the network has
            layer_kernel_sizes (list[tuple[int, int]]): A list of the kernel
                sizes for each layer, the length of this list should be the
                same as the length of layer_depths
            num_classes (int): How many classes the network should classify
                affects the output size of the call
            input_size (int): The size of the input image, the image should be
                square, e.g. 28 for MNIST
            num_hidden_layers (int): The number of hidden layers in the MLP
            hidden_layer_width (int): The width of the hidden layers in the MLP
            pool_every_n_layers (int, optional): Adds a max pooling layer
                every n layers. Defaults to 0. Aka, no
                pooling layers.
            pool_size (int, optional): The size of the kernel for the max
                pooling layer. Defaults to 2.
            dropout_prob (float, optional): The probability of dropping a node
            group_norm_num_groups (int, optional): The number of groups to split the
                channels into for group normalization. Defaults to 32.
        """
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.input_size = input_size
        self.pool_size = pool_size
        self.resblock_size = resblock_size
        self.dropout_prob = dropout_prob

        output_depth = self.layer_depths[-1]
        self.flatten_size = int(
            (input_size // (self.pool_size ** (2))) ** 2 * output_depth
        )

        self.conv_layers = []
        self.shortcut_conv_layers = []
        self.group_norm_layers = []
        for layer_depth, layer_kernel_size, group_norm_num_group in zip(
            self.layer_depths, self.layer_kernel_sizes, group_norm_num_groups
        ):
            self.conv_layers.append(Conv2D(input_depth, layer_depth, layer_kernel_size))

            self.shortcut_conv_layers.append(
                Conv2D(
                    input_depth,
                    layer_depth,
                    [1, 1],
                )
            )

            self.group_norm_layers.append(GroupNorm(group_norm_num_group, layer_depth))
            input_depth = layer_depth

            self.fully_connected = MLP(
                self.flatten_size,
                num_classes,
                num_hidden_layers,
                hidden_layer_width,
                hidden_activation=tf.nn.relu,
                zero_init=True,
            )

    def __call__(self, input_tensor: tf.Tensor):
        """Applies the classifier to the input,
             runs it through a series of convolutional layers which consists of
             a convolution, a relu, and a max pooling layer every n layers
             then flattens the output and runs it through a linear layer,
             then sends it through a softmax activation

        Args:
            input_tensor (tf.Tensor): The Image to classify, should have shape
                [batch_size, input_size, input_size, input_depth]

        Returns:
            tf.Tensor: The logits of the classification, should have shape
                [batch_size, num_classes]
        """
        moving_input_tensor = input_tensor

        moving_input_tensor = tf.nn.max_pool2d(
            moving_input_tensor, self.pool_size, strides=2, padding="VALID"
        )

        # TODO: START OF RESBLOCK CALL
        for conv_layer, shortcut_layer, group_norm_layer in zip(
            self.conv_layers, self.shortcut_conv_layers, self.group_norm_layers
        ):
            shortcut = shortcut_layer(moving_input_tensor)

            for _ in range(self.resblock_size):
                output_tensor = conv_layer(moving_input_tensor)

                output_tensor = group_norm_layer(output_tensor)
                output_tensor = tf.nn.relu(output_tensor)

            moving_input_tensor = output_tensor + shortcut
        # TODO: END OF RESBLOCK CALL

        output_tensor = tf.nn.avg_pool2d(
            moving_input_tensor, self.pool_size, strides=2, padding="VALID"
        )

        output_tensor = tf.nn.dropout(output_tensor, rate=self.dropout_prob)

        if self.flatten_size != (
            output_tensor.shape[1] * output_tensor.shape[2] * output_tensor.shape[3]
        ):
            raise ValueError("Flatten size does not match output tensor shape")

        output_flattened = tf.reshape(output_tensor, [-1, self.flatten_size])

        output_tensor = self.fully_connected(output_flattened)
        return output_tensor
