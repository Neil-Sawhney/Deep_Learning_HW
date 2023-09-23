import tensorflow as tf


def augment_data(images: tf.Tensor, augmentations: int):
    """Takes images with n number of channels (e.g. 3 for RGB),
      and applies random augmentations to make it have augmentations*n number
      of channels (e.g. 12 for RGB and 4 augmentations)
      applies augmentation_per_type augmentations of each type
      (e.g. rotation, translation, etc.)
        

    Args:
        images (tf.Tensor): should be a 4D tensor of shape
            [batch_size, height, width, channels]

    Returns:
        tf.Tensor: A tensor of the shape
        [batch_size, height, width, channels*augmentations]
    """
    augmentations_per_type = augmentations // 4
    # apply augmentations_per_type rotations of various degrees