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


@pytest.mark.parametrize("num_bases", [1, 16, 128])
def test_dimensionality(num_bases):
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_outputs = 10
    num_inputs = 5

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)

    a = rng.normal(shape=[num_inputs, 1])
    z = basisExpansion(a)
    tf.assert_equal(tf.shape(z)[0], num_inputs)
    tf.assert_equal(tf.shape(z)[-1], num_bases)


def test_trainable():
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 10
    num_bases = 20

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)

    a = rng.normal(shape=[num_inputs, 1])

    with tf.GradientTape() as tape:
        z = basisExpansion(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, basisExpansion.trainable_variables)

    for grad, var in zip(grads, basisExpansion.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.Assert(
            tf.reduce_any(
                tf.greater(
                    tf.abs(
                        grad
                        ),
                    0
                    )
                ),
            [grad],
            summarize=2
        )

    assert len(grads) == len(basisExpansion.trainable_variables)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ([1000, 1000], [100, 100]),
        ([1000, 100], [100, 100]),
        ([100, 1000], [100, 100])
    ],
)
def test_init_properties(a_shape, b_shape):
    import tensorflow as tf

    from basisExpansion import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape
    num_outputs = 10

    linear_a = BasisExpansion(num_outputs_a, num_inputs_a, num_outputs)
    linear_b = BasisExpansion(num_outputs_b, num_inputs_b, num_outputs)

    std_a = tf.math.reduce_std(linear_a.mu)
    std_b = tf.math.reduce_std(linear_b.mu)
    std_c = tf.math.reduce_std(linear_a.sigma)
    std_d = tf.math.reduce_std(linear_b.sigma)

    tf.debugging.assert_less(std_a, std_b)
    tf.debugging.assert_less(std_c, std_d)


if __name__ == "__main__":
    pytest.main([__file__])
