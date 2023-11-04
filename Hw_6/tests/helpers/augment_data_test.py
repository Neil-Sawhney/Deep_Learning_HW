import pytest


@pytest.mark.parametrize("augmentation_probability", [0.0, 0.5, 1.0])
def test_dimenionality(augmentation_probability):
    import tensorflow as tf

    from helpers.augment_data import AugmentData
    from helpers.load_pickle_data import load_pickle_data

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    augment_data = AugmentData(augmentation_probability)

    labels, images = load_pickle_data("data/cifar-10-batches-py/data_batch_1")

    augmented_labels, augmented_images = augment_data(labels, images)

    # check the image dimensions
    assert (
        tf.shape(augmented_images)[0].numpy()
        == tf.shape(images)[0].numpy()
        + augmentation_probability * tf.shape(images)[0].numpy() * 6
    )

    # check the label dimensions
    assert (
        tf.shape(augmented_labels)[0].numpy()
        == tf.shape(labels)[0].numpy()
        + augmentation_probability * tf.shape(labels)[0].numpy() * 6
    )


if __name__ == "__main__":
    pytest.main([__file__])
