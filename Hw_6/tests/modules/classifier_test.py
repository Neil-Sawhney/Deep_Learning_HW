import pytest


@pytest.mark.parametrize("layer_depths", [[10, 20], [20, 40]])
@pytest.mark.parametrize("kernel_sizes", [[[2, 2], [2, 2]], [[3, 3], [3, 3]]])
@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("resblock_size", [1, 2, 3])
@pytest.mark.parametrize("group_norm_num_groups", [[2, 2], [2, 5]])
def test_dimensionality(
    layer_depths,
    kernel_sizes,
    num_classes,
    resblock_size,
    group_norm_num_groups,
):
    import tensorflow as tf

    from modules.classifier import Classifier

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_depth = 3
    input_size = 32
    dropout_prob = 0.5
    pool_size = 2

    num_hidden_layers = 1
    hidden_layer_width = 10

    classifier = Classifier(
        input_depth,
        layer_depths,
        kernel_sizes,
        num_classes,
        input_size,
        resblock_size,
        pool_size,
        dropout_prob,
        group_norm_num_groups,
        num_hidden_layers,
        hidden_layer_width,
    )

    input_data = rng.normal(shape=[1, input_size, input_size, input_depth])
    classified_data = classifier(input_data)

    tf.assert_equal(tf.shape(classified_data)[-1], num_classes)


if __name__ == "__main__":
    pytest.main([__file__])
