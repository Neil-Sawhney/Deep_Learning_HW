import argparse
from pathlib import Path

import tensorflow as tf
import yaml
from tqdm import trange

from helpers.idx_loader import load_idx_data
from helpers.optimizer import Adam
from models.classifier import Classifier


def train_batch_accuracy():
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(train_images_batch).numpy().argmax(axis=1),
                train_labels_batch.numpy().reshape(-1)
            ),
            tf.float32)
    )


def val_accuracy():
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(val_images).numpy().argmax(axis=1),
                val_labels.numpy().reshape(-1)
            ),
            tf.float32)
    )


def test_accuracy():
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(test_images).numpy().argmax(axis=1),
                test_labels.numpy().reshape(-1)
            ),
            tf.float32)
    )


def val_loss():
    global minimum_val_loss, checkpoint
    validation_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(val_labels),
            logits=classifier(val_images)
        )
    )

    if validation_loss < minimum_val_loss:
        minimum_val_loss = validation_loss
        checkpoint.save(
            "artifacts/checkpoints/"
        )

    return validation_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CNN",
        description="Uses a multi layer perceptron on some data, \
                given a config",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml")
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    train_images = tf.cast(
        load_idx_data("data/train-images-idx3-ubyte")[:-10000], tf.float32)
    train_labels = tf.cast(
        load_idx_data("data/train-labels-idx1-ubyte")[:-10000], tf.int32)
    val_images = tf.cast(
        load_idx_data("data/train-images-idx3-ubyte")[-10000:], tf.float32)
    val_labels = tf.cast(
        load_idx_data("data/train-labels-idx1-ubyte")[-10000:], tf.int32)
    test_images = tf.cast(
        load_idx_data("data/t10k-images-idx3-ubyte"), tf.float32)
    test_labels = tf.cast(
        load_idx_data("data/t10k-labels-idx1-ubyte"), tf.int32)

    layer_depths = config["cnn"]["layer_depths"]
    kernel_sizes = config["cnn"]["kernel_sizes"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    dropout_prob = config["learning"]["dropout_prob"]
    batch_size = config["learning"]["batch_size"]
    refresh_rate = config["display"]["refresh_rate"]
    pool_every_n_layers = config["cnn"]["pool_every_n_layers"]
    pool_size = config["cnn"]["pool_size"]
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]
    learning_rate = config["learning"]["learning_rate"]

    num_samples = train_images.shape[0]
    input_depth = train_images.shape[-1]
    classifier = Classifier(input_depth,
                            layer_depths,
                            kernel_sizes,
                            10,
                            train_images.shape[1],
                            num_hidden_layers,
                            hidden_layer_width,
                            pool_every_n_layers,
                            pool_size,
                            dropout_prob)

    bar = trange(num_iters)
    adam = Adam(learning_rate, weight_decay=weight_decay)

    minimum_val_loss = float("inf")

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:

            train_images_batch = tf.gather(train_images, batch_indices)
            train_labels_batch = tf.gather(train_labels, batch_indices)
            kernel_weights = tf.concat(
                [tf.reshape(layer.kernel, [-1])
                 for layer in classifier.conv_layers], axis=0)
            training_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(train_labels_batch),
                    logits=classifier(train_images_batch)
                ))

        grads = tape.gradient(training_loss, classifier.trainable_variables)

        adam.apply_gradients(
            grads, classifier.trainable_variables
        )

        checkpoint = tf.train.Checkpoint(classifier)

        validation_loss = val_loss()

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {training_loss.numpy():0.4f};" +
                f" Train Batch Accuracy => {train_batch_accuracy():0.4};" +
                f" Val Accuracy => {val_accuracy():0.4f}" +
                f" Val loss => {validation_loss:0.4f}"
            )
            bar.refresh()

        if validation_loss > minimum_val_loss + 0.2:
            break

    checkpoint.restore(tf.train.latest_checkpoint("artifacts/checkpoints/"))

    print(f"Test Accuracy => {test_accuracy():0.4f}")
