import tensorflow as tf


class Classifier(tf.Module):
    def __init__(
            self,
            input_depth: int,
            layer_depths: list[int],
            layer_kernel_sizes: list[tuple[int, int]],
            num_classes: int,
            ):
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes

        self.conv_layers = []
        self.conv_layers.append(tf.keras.layers.Conv2D(self.layer_depths[0], self.layer_kernel_sizes[0], activation=tf.nn.relu, input_shape=(28, 28, 1)))
        for i in range(1, len(self.layer_depths)):
            self.conv_layers.append(tf.keras.layers.Conv2D(self.layer_depths[i], self.layer_kernel_sizes[i], activation=tf.nn.relu))
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)

    def __call__(self, x):
        x = self.conv_layers[0](x)
        for i in range(1, len(self.conv_layers)):
            x = self.conv_layers[i](x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x