import tensorflow as tf
from conv2d import Conv2D
from linear import Linear


class Classifier(tf.Module):
    def __init__(
            self,
            input_depth: int,
            layer_depths: list[int],
            layer_kernel_sizes: list[tuple[int, int]],
            num_classes: int,
            input_size: int,
            pool_every_n_layers: int = 0,
            pool_size: int = 2,
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
            pool_every_n_layers (int, optional): Adds a max pooling layer
                every n layers. The number of layers specified by layer_depths
                should be divisible by this number. Defaults to 0. Aka, no
                pooling layers.
            pool_size (int, optional): The size of the kernel for the max
                pooling layer. Defaults to 2.
        """
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        self.input_size = input_size
        self.pool_every_n_layers = pool_every_n_layers
        self.pool_size = pool_size

        # TODO: Make sure the flatten size is correct
        num_layers = self.layer_depths.__len__()
        if self.pool_every_n_layers > 0:
            num_pools = num_layers // self.pool_every_n_layers
            self.flatten_size = (
                (input_size / (self.pool_size ** (num_pools)))**2 *
                num_layers)
        else:
            self.flatten_size = (input_size**2 * num_layers)

        self.conv_layers = []
        for (layer_depth,
             layer_kernel_size) in zip(
                 self.layer_depths,
                 self.layer_kernel_sizes):
            self.conv_layers.append(Conv2D(self.input_depth,
                                           layer_depth,
                                           layer_kernel_size))
            self.input_depth = self.layer_depth

        self.linear = Linear(self.flatten_size,
                             self.num_classes)

    def __call__(self, input: tf.Tensor):
        """Applies the classifier to the input,
             runs it through a series of convolutional layers which consists of
             a convolution, a relu, and a max pooling layer every n layers
             then flattens the output and runs it through a linear layer,
             then sends it through a softmax activation

        Args:
            input (tf.Tensor): The Image to classify, should have shape
                [batch_size, input_size, input_size, input_depth]

        Returns:
            tf.Tensor: The logits of the classification, should have shape
                [batch_size, num_classes]
                each logit represents the confidence of the network that the
                image belongs to that class
        """
        for conv_layer in self.conv_layers:
            input = tf.nn.relu(conv_layer(input))
            if self.pool_every_n_layers > 0:
                input = tf.nn.max_pool2d(input,
                                         self.pool_size,
                                         self.pool_size,
                                         "VALID")

        # TODO: Make sure the flatten is correct
        input_flattened = tf.reshape(input, [-1, self.flatten_size])
        return tf.nn.softmax(self.linear(input_flattened))
