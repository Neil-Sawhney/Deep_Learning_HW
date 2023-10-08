import tensorflow as tf


class AugmentData:
    def __init__(self, augmentation_prob: int):
        """Initializes the AugmentData class

        Args:
            augmentation_prob (int): the probability of augmenting the data, from 0 to 1
        """
        self.augmentation_prob = augmentation_prob

    def __call__(
        self, labels: tf.Tensor, images: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Takes the data and augments it,
           by applying random zooms, rotations, brightness, contrast, hue, and
           saturation
        Args:
            images (tf.Tensor): a tensor of RGB images of shape
            [batch_size, height, width, 3]
            labels (tf.Tensor): a tensor of labels of shape [batch_size]
        Returns:
            tuple[tf.Tensor, tf.Tensor]: a tuple of the augmented images and labels,
            the images have shape [batch_size +
            augmentation_prob*batch_size*6, height, width, 3] and the labels have
            shape [batch_size + augmentation_prob*batch_size*6]
        """

        zoom_indices = tf.random.uniform(
            shape=[
                tf.cast(tf.shape(images)[0].numpy() * self.augmentation_prob, tf.int32)
            ],
            maxval=tf.shape(images)[0],
            dtype=tf.int32,
        )
        flipped_indices = tf.random.uniform(
            shape=[
                tf.cast(tf.shape(images)[0].numpy() * self.augmentation_prob, tf.int32)
            ],
            maxval=tf.shape(images)[0],
            dtype=tf.int32,
        )
        brightness_indices = tf.random.uniform(
            shape=[
                tf.cast(tf.shape(images)[0].numpy() * self.augmentation_prob, tf.int32)
            ],
            maxval=tf.shape(images)[0],
            dtype=tf.int32,
        )
        contrast_indices = tf.random.uniform(
            shape=[
                tf.cast(tf.shape(images)[0].numpy() * self.augmentation_prob, tf.int32)
            ],
            maxval=tf.shape(images)[0],
            dtype=tf.int32,
        )
        hue_indices = tf.random.uniform(
            shape=[
                tf.cast(tf.shape(images)[0].numpy() * self.augmentation_prob, tf.int32)
            ],
            maxval=tf.shape(images)[0],
            dtype=tf.int32,
        )
        saturation_indices = tf.random.uniform(
            shape=[
                tf.cast(tf.shape(images)[0].numpy() * self.augmentation_prob, tf.int32)
            ],
            maxval=tf.shape(images)[0],
            dtype=tf.int32,
        )

        # apply augmentations to the images
        zoomed_images = tf.gather(images, zoom_indices)
        zoomed_images = tf.image.random_crop(
            zoomed_images, [zoomed_images.shape[0], 24, 24, 3]
        )
        zoomed_images = tf.image.resize(zoomed_images, [32, 32])

        flipped_images = tf.gather(images, flipped_indices)
        flipped_images = tf.image.random_flip_left_right(flipped_images)

        brightness_images = tf.gather(images, brightness_indices)
        brightness_images = tf.image.random_brightness(brightness_images, 0.2)

        contrast_images = tf.gather(images, contrast_indices)
        contrast_images = tf.image.random_contrast(contrast_images, 0.2, 0.5)

        hue_images = tf.gather(images, hue_indices)
        hue_images = tf.image.random_hue(hue_images, 0.2)

        saturation_images = tf.gather(images, saturation_indices)
        saturation_images = tf.image.random_saturation(saturation_images, 0.2, 0.5)

        # combine the augmented images with the original images
        output_images = tf.concat(
            [
                images,
                zoomed_images,
                flipped_images,
                brightness_images,
                contrast_images,
                hue_images,
                saturation_images,
            ],
            axis=0,
        )

        output_labels = tf.concat(
            [
                labels,
                tf.gather(labels, zoom_indices),
                tf.gather(labels, flipped_indices),
                tf.gather(labels, brightness_indices),
                tf.gather(labels, contrast_indices),
                tf.gather(labels, hue_indices),
                tf.gather(labels, saturation_indices),
            ],
            axis=0,
        )

        return output_labels, output_images
