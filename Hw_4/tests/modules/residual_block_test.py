import pytest


@pytest.mark.parametrize("input_depth", [1, 3])
@pytest.mark.parametrize("output_depth", [1, 3])
@pytest.mark.parametrize("kernel_size", [[2, 2], [4, 4], [8, 8]])
@pytest.mark.parametrize("group_norm_num_groups", [1, 3, 16])
@pytest.mark.parametrize("resblock_size", [1, 2, 3])
def test_dimensionality(
    input_depth, output_depth, kernel_size, group_norm_num_groups, resblock_size
):
    import tensorflow as tf

    from modules.residual_block import ResidualBlock

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    residual_block = ResidualBlock(
        input_depth, output_depth, kernel_size, group_norm_num_groups, resblock_size
    )

    input_tensor = rng.normal(shape=[1, 32, 32, input_depth])
    output_tensor = residual_block(input_tensor)

    tf.assert_equal(tf.shape(output_tensor)[-1], output_depth)
    tf.assert_equal(tf.shape(output_tensor)[0:3], tf.shape(input_tensor)[0:3])


if __name__ == "__main__":
    pytest.main([__file__])
