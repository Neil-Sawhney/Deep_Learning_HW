import tensorflow as tf

from modules.linear import Linear


class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
        dropout_prob=0,
        zero_init=False,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_linear = Linear(self.hidden_layer_width, self.hidden_layer_width)
        self.first_linear = Linear(num_inputs, hidden_layer_width)
        self.final_linear = Linear(
            self.hidden_layer_width, self.num_outputs, zero_init=zero_init
        )
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        """Applies the MLP to the input

        Args:
            x (tf.tensor): input tensor of shape [batch_size, num_inputs]

        Returns:
            tf.tensor: output tensor of shape [batch_size, num_outputs]
        """
        x = self.hidden_activation(self.first_linear(x))

        for _ in range(self.num_hidden_layers):
            x = self.hidden_activation(self.hidden_linear(x))

        if self.dropout_prob > 0:
            x = tf.nn.dropout(x, self.dropout_prob)

        return self.output_activation(self.final_linear(x))
