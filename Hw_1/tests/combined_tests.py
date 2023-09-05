# /bin/env python3.8

import pytest


def test_additivity():
    import tensorflow as tf

    from basisExpansion import BasisExpansion
    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_bases = 10

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)
    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(
        linear(basisExpansion(a) + basisExpansion(b)),
        linear(basisExpansion(a)) + linear(basisExpansion(b)), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from basisExpansion import BasisExpansion
    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100
    num_bases = 10

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)
    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1, 1])

    tf.debugging.assert_near(
        linear(basisExpansion(a) * b),
        linear(basisExpansion(a)) * b,
        summarize=2
        )


if __name__ == "__main__":
    pytest.main([__file__])
