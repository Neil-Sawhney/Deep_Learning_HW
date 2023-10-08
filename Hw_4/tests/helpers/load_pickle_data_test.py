import pytest


def test_dimensionality():
    import tensorflow as tf

    from helpers.load_pickle_data import load_pickle_data

    train_and_val_labels, train_and_val_images = load_pickle_data(
        "data/cifar-10-batches-py/data_batch_1"
    )

    assert tf.shape(train_and_val_labels) == [10000]
    assert tf.shape(train_and_val_images) == [10000, 32, 32, 3]

    train_and_val_labels, train_and_val_images = load_pickle_data(
        "data/cifar-100-python/train"
    )

    assert tf.shape(train_and_val_labels) == [50000]
    assert tf.shape(train_and_val_images) == [50000, 32, 32, 3]


def test_labels():
    import tensorflow as tf

    from helpers.load_pickle_data import load_pickle_data

    train_and_val_labels, train_and_val_images = load_pickle_data(
        "data/cifar-10-batches-py/data_batch_1"
    )

    assert tf.reduce_min(train_and_val_labels) == 0
    assert tf.reduce_max(train_and_val_labels) == 9


if __name__ == "__main__":
    pytest.main([__file__])
