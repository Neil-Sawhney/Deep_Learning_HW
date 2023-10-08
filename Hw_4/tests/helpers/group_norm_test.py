import pytest


def test_dimensionality():
    import tensorflow as tf
    from helpers.group_norm import GroupNorm

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    group_norm = GroupNorm(4, 16)

    input = rng.normal(shape=[1, 32, 32, 16])
    output = group_norm(input)

    tf.assert_equal(tf.shape(output), tf.shape(input))


def test_mean_and_variance():
    import tensorflow as tf
    from helpers.group_norm import GroupNorm

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    group_norm = GroupNorm(4, 16)

    input = rng.normal(shape=[1, 32, 32, 16])
    output = group_norm(input)

    mean, variance = tf.nn.moments(output, axes=[1, 2, 3])

    tf.assert_equal(tf.shape(mean), [1])
    tf.assert_equal(tf.shape(variance), [1])

    tf.assert_equal(mean, tf.zeros_like(mean))
    tf.assert_equal(variance, tf.ones_like(variance))


if __name__ == "__main__":
    pytest.main([__file__])
