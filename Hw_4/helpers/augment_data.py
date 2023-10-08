import tensorflow as tf


class AugmentData:
    def __init__(self, augmentation_prob: int, num_augmentation_attempts: int = 1):
        """Initializes the AugmentData class

        Args:
            augmentation_prob (int): the probability of augmenting the data, from 0 to 1
            num_augmentation_attempts (int, optional): the number of
            times to attempt to augment the data. Defaults to 1.
        """
        self.augmentation_prob = augmentation_prob
        self.num_augmentation_attempts = num_augmentation_attempts

    def __call__(self, images: tf.Tensor):
        """Takes the data and augments it,
           by applying random zooms, rotations, brightness, contrast, hue, and
           saturation
        Args:
            images (tf.Tensor): a tensor of images of shape
            [batch_size, height, width, channels]
        Returns:
            tf.Tensor: a tensor of images and augmented images of shape
            [batch_size * num_of_augmentations + batch_size, height, width, channels]
        """
        output = []
        for image in images:
            for _ in range(self.num_augmentation_attempts):
                if tf.random.uniform(()).numpy() > self.augmentation_prob:
                    continue
                random_number = tf.random.uniform(
                    (), minval=0, maxval=6, dtype=tf.int32
                )
                # 0 = zoom, 1 = flip, 2 = brightness,
                # 3 = contrast, 4 = hue, 5 = saturation
                if random_number == 0:
                    image = tf.image.random_crop(image, [24, 24, 3])
                    image = tf.image.resize(image, [32, 32])
                elif random_number == 1:
                    image = tf.image.random_flip_left_right(image)
                elif random_number == 2:
                    image = tf.image.random_brightness(image, max_delta=0.1)
                elif random_number == 3:
                    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
                elif random_number == 4:
                    image = tf.image.random_hue(image, max_delta=0.1)
                elif random_number == 5:
                    image = tf.image.random_saturation(image, lower=0.1, upper=0.2)
            output.append(image)
        return tf.stack(output)
