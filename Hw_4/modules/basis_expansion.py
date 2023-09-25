import tensorflow as tf


class BasisExpansion(tf.Module):
    def __init__(self, num_bases, num_inputs, num_outputs):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.mu = tf.Variable(
            rng.normal(shape=[num_inputs, num_bases], stddev=stddev),
            trainable=True,
            name="BasisExpansion/mu",
        )

        self.sigma = tf.Variable(
            rng.normal(shape=[num_inputs, num_bases], stddev=stddev),
            trainable=True,
            name="BasisExpansion/sigma",
        )

    def __call__(self, x):
        z = tf.exp(-(x - self.mu)**2 / (self.sigma**2))

        return z
