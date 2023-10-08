import pytest


def test_dimensionality():
    from helpers.load_idx_data import load_idx_data

    train_images = load_idx_data("data/train-images-idx3-ubyte")
    train_labels = load_idx_data("data/train-labels-idx1-ubyte")
    test_labels = load_idx_data("data/t10k-labels-idx1-ubyte")
    test_images = load_idx_data("data/t10k-images-idx3-ubyte")

    assert train_images.shape == (60000, 28, 28, 1)
    assert train_labels.shape == (60000, 1)
    assert test_images.shape == (10000, 28, 28, 1)
    assert test_labels.shape == (10000, 1)


if __name__ == "__main__":
    pytest.main([__file__])
