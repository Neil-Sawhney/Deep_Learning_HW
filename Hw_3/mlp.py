import tensorflow as tf

from linear import Linear


class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers,
                 hidden_layer_width, hidden_activation=tf.identity,
                 output_activation=tf.identity):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_linear = Linear(self.hidden_layer_width,
                                    self.hidden_layer_width)
        self.first_linear = Linear(num_inputs, hidden_layer_width)
        self.final_linear = Linear(self.hidden_layer_width, self.num_outputs)

    def __call__(self, x):
        x = self.hidden_activation(self.first_linear(x))
        for i in range(self.num_hidden_layers):
            x = self.hidden_activation(
                self.hidden_linear(x)
            )
        return self.output_activation(self.final_linear(x))
