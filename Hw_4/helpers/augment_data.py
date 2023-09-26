import tensorflow as tf


def augment_data(images: tf.Tensor, num_augmentations: int):
    """Takes the data and augments it,
       by applying random flips, rotations, brightness, contrast, hue, and saturation
    Args:
        images (tf.Tensor): a tensor of images of shape
        [batch_size, height, width, channels]
        num_of_augmentations (int): the number of augmentations to apply to each image
    Returns:
        tf.Tensor: a tensor of images and augmented images of shape
        [batch_size * num_of_augmentations + batch_size, height, width, channels]
    """
    for _ in range(num_augmentations):
        tf.image.random_flip_left_right(images),
        tf.image.random_flip_up_down(images),
        tf.image.rot90(images, k=1),
        tf.image.rot90(images, k=2),
        tf.image.rot90(images, k=3),
        tf.image.random_brightness(images, max_delta=0.1),
        tf.image.random_contrast(images, lower=0.1, upper=0.2),
        tf.image.random_hue(images, max_delta=0.1),
        tf.image.random_saturation(images, lower=0.1, upper=0.2),

    return images
