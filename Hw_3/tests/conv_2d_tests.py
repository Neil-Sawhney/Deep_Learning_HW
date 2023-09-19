import pytest


@pytest.mark.parametrize("kernel_size", [2, 4, 8])
@pytest.mark.parametrize("input_channels", [1, 3, 16])
@pytest.mark.parametrize("output_channels", [1, 3, 16])
def test_dimensionality(kernel_size, input_channels, output_channels):
    import tensorflow as tf
    from functional.conv_2d import Conv2D

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    conv2d = Conv2D(input_channels, output_channels, kernel_size)

    a = rng.normal(shape=[1, 28, 28, input_channels])
    z = conv2d(a)

    tf.assert_equal(tf.shape(z)[-1], output_channels)


if __name__ == "__main__":
    pytest.main([__file__])
