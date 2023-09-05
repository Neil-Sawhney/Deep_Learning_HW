# /bin/env python3.8

import pytest


def test_non_additivity():
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_bases = 10

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    case1 = basisExpansion(a + b)
    case2 = basisExpansion(a) + basisExpansion(b)

    tol = 2.22e-15 + 2.22e-15*tf.abs(case2)

    tf.debugging.Assert(
        tf.reduce_any(
            tf.greater(
                tf.abs(
                    case1 - case2
                    ),
                tol
                )
            ),
        [case1, case2],
        summarize=2
    )


def test_homogeneity():
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_bases = 10
    num_test_cases = 100

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)

    a = rng.normal(shape=[1, 1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1, 1])

    case1 = basisExpansion(a * b)
    case2 = basisExpansion(a) * b

    tol = 2.22e-15 + 2.22e-15*tf.abs(case2)

    tf.debugging.Assert(
        tf.reduce_any(
            tf.greater(
                tf.abs(
                    case1 - case2
                    ),
                tol
                )
            ),
        [case1, case2],
        summarize=2
    )


@pytest.mark.parametrize("num_inputs", [1, 16, 128])
def test_dimensionality(num_inputs):
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_bases = 10
    num_outputs = 5

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)

    a = rng.normal(shape=[num_inputs, 1])
    z = basisExpansion(a)
    tf.assert_equal(tf.shape(z)[0], num_inputs)


if __name__ == "__main__":
    pytest.main([__file__])
