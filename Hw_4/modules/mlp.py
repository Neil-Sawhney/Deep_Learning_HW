import tensorflow as tf

from modules.linear import Linear


class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers,
                 hidden_layer_width, hidden_activation=tf.identity,
                 output_activation=tf.identity,
                 dropout_first_n_layers=0,
                 dropout_prob=0.5):
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
        self.dropout_first_n_layers = dropout_first_n_layers
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        if self.dropout_first_n_layers >= 1:
            x = tf.nn.dropout(x, self.dropout_prob)

        x = self.hidden_activation(self.first_linear(x))

        if self.dropout_first_n_layers >= 2:
            x = tf.nn.dropout(x, self.dropout_prob)

        for i in range(self.num_hidden_layers):
            if self.dropout_first_n_layers > i + 2:
                x = tf.nn.dropout(x, self.dropout_prob)
            x = self.hidden_activation(
                self.hidden_linear(x)
            )
        return self.output_activation(self.final_linear(x))
