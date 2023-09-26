import tensorflow as tf


class Linear(tf.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 bias=True,
                 zero_init=False,
                 identity=False):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.bias = bias

        if identity:
            self.w = tf.Variable(
                tf.eye(num_inputs, num_outputs),
                trainable=True,
                name="Linear/w",
            )

        elif zero_init:
            self.w = tf.Variable(
                tf.zeros(shape=[num_inputs, num_outputs]),
                trainable=True,
                name="Linear/w",
            )
        else:
            self.w = tf.Variable(
                rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
                trainable=True,
                name="Linear/w",
            )

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    # create the logits by multiplying the inputs by the weights + the
    # optional bias
    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z
