# /bin/env python3.8

import pytest


def test_additivity():
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    basisExpansion = BasisExpansion(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    # assert that the basisExpansion(a + b) is not near basisExpansion(a) + basisExpansion(b)
    tf.debugging.assert_near(basisExpansion(a + b), basisExpansion(a) + basisExpansion(b), summarize=2)