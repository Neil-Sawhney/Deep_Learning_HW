import tensorflow as tf
from functional.conv_2d import Conv2D
from models.mlp import MLP


class Classifier(tf.Module):
    def __init__(
            self,
            input_depth: int,
            layer_depths: list[int],
            layer_kernel_sizes: list[tuple[int, int]],
            num_classes: int,
            input_size: int,
            num_hidden_layers: int,
            hidden_layer_width: int,
            pool_every_n_layers: int = 0,
            pool_size: int = 2,
            dropout_prob: float = 0.5,
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
        """
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.pool_every_n_layers = pool_every_n_layers
        self.pool_size = pool_size
        self.dropout_prob = dropout_prob

        num_layers = len(self.layer_depths)
        output_depth = self.layer_depths[-1]
        if self.pool_every_n_layers > 0:
            num_pools = num_layers // self.pool_every_n_layers
            self.flatten_size = int(
                (input_size / (self.pool_size ** (num_pools)))**2 *
                output_depth)
        else:
            self.flatten_size = int(input_size**2 * output_depth)

        self.conv_layers = []
        for (layer_depth,
             layer_kernel_size) in zip(
                 self.layer_depths,
                 self.layer_kernel_sizes):
            self.conv_layers.append(Conv2D(input_depth,
                                           layer_depth,
                                           layer_kernel_size))
            input_depth = layer_depth

        self.mlp = MLP(self.flatten_size,
                       self.num_classes,
                       self.num_hidden_layers,
                       self.hidden_layer_width,
                       tf.nn.relu)

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
        for i, conv_layer in enumerate(self.conv_layers):
            input_tensor = tf.nn.relu(conv_layer(input_tensor))
            input_tensor = tf.nn.dropout(input_tensor, self.dropout_prob)
            if self.pool_every_n_layers > 0:
                if (i + 1) % self.pool_every_n_layers == 0:
                    input_tensor = tf.nn.max_pool2d(input_tensor,
                                                    self.pool_size,
                                                    self.pool_size,
                                                    "VALID")
        input_flattened = tf.reshape(input_tensor, [-1, self.flatten_size])
        return self.mlp(input_flattened)
