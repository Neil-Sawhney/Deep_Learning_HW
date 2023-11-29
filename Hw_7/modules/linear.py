import tensorflow as tf


class Linear(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        bias=True,
        zero_init=False,
        siren_init=False,
    ):
        rng = tf.random.get_global_generator()

        stddev = tf.cast(tf.math.sqrt(2 / (num_inputs + num_outputs)), tf.float32)

        self.bias = bias

        w_initial_value = rng.normal(shape=[num_inputs, num_outputs], stddev=stddev)
        if zero_init:
            w_initial_value = tf.zeros(shape=[num_inputs, num_outputs])
        elif siren_init:
            w_initial_value = rng.uniform(
                minval=-tf.math.sqrt(6 / num_inputs),
                maxval=tf.math.sqrt(6 / num_inputs),
                shape=[num_inputs, num_outputs],
            )
        self.w = tf.Variable(
            w_initial_value,
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
