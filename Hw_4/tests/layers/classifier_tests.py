import pytest


@pytest.mark.parametrize("input_depth", [1, 3])
@pytest.mark.parametrize("layer_depths", [[1, 1], [3, 3]])
@pytest.mark.parametrize("layer_kernel_sizes", [[[2,2], [2,2]],
                                                [[4,4], [4,4]]])
@pytest.mark.parametrize("num_classes", [1, 3])
@pytest.mark.parametrize("input_size", [16, 32, 64])
@pytest.mark.parametrize("pool_every_n_layers", [0, 1, 2])
@pytest.mark.parametrize("pool_size", [2, 4])
def test_dimensionality(input_depth,
                        layer_depths,
                        layer_kernel_sizes,
                        num_classes,
                        input_size,
                        pool_every_n_layers,
                        pool_size):
    import tensorflow as tf
    from layers.classifier import Classifier

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_hidden_layers = 2
    hidden_layer_width = 20

    classifier = Classifier(input_depth,
                            layer_depths,
                            layer_kernel_sizes,
                            num_classes,
                            input_size,
                            num_hidden_layers,
                            hidden_layer_width,
                            pool_every_n_layers,
                            pool_size)

    input_data = rng.normal(shape=[1, input_size, input_size, input_depth])
    classified_data = classifier(input_data)

    tf.assert_equal(tf.shape(classified_data)[-1], num_classes)


if __name__ == "__main__":
    pytest.main([__file__])
