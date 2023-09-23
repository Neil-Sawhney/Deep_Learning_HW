from pathlib import Path

import tensorflow as tf
import yaml
from tqdm import trange

from helpers.idx_loader import load_idx_data
from helpers.optimizer import Adam
from models.classifier import Classifier


def train_batch_accuracy(classifier,
                         train_images_batch,
                         train_labels_batch):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(train_images_batch).numpy().argmax(axis=1),
                train_labels_batch.numpy().reshape(-1)
            ),
            tf.float32)
    )


def val_accuracy(classifier,
                 val_images,
                 val_labels):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(val_images).numpy().argmax(axis=1),
                val_labels.numpy().reshape(-1)
            ),
            tf.float32)
    )


def test_accuracy(classifier,
                  test_images,
                  test_labels):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(test_images).numpy().argmax(axis=1),
                test_labels.numpy().reshape(-1)
            ),
            tf.float32)
    )


def val_loss(classifier,
             val_images,
             val_labels,
             checkpoint,
             minimum_val_loss,
             minimum_val_step,
             current_step):
    validation_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(val_labels),
            logits=classifier(val_images)
        )
    )

    if validation_loss < minimum_val_loss:
        minimum_val_loss = validation_loss
        minimum_val_step = current_step
        checkpoint.write(
            "artifacts/checkpoints/classify_numbers"
        )

    return validation_loss, minimum_val_loss, minimum_val_step


def run(config_path: Path = Path("configs/classify_numbers_config.yaml")):

    config = yaml.safe_load(config_path.read_text())

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
    learning_patience = config["learning"]["learning_patience"]
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
    minimum_val_step = 0
    current_validation_loss = 0
    checkpoint = tf.train.Checkpoint(classifier)
    tf.train.CheckpointManager(
        checkpoint, "artifacts/checkpoints/classify_numbers", max_to_keep=1
    )

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:

            train_images_batch = tf.gather(train_images, batch_indices)
            train_labels_batch = tf.gather(train_labels, batch_indices)
            current_training_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(train_labels_batch),
                    logits=classifier(train_images_batch)
                ))

        grads = tape.gradient(current_training_loss,
                              classifier.trainable_variables)

        adam.apply_gradients(
            grads, classifier.trainable_variables
        )

        # If no improvement in validation loss for learning_patience
        # iterations, stop training
        current_validation_loss, minimum_val_loss, minimum_val_step = \
            val_loss(classifier,
                     val_images,
                     val_labels,
                     checkpoint,
                     minimum_val_loss,
                     minimum_val_step,
                     i)

        if i % refresh_rate == (refresh_rate - 1):
            current_training_loss = current_training_loss.numpy()
            currect_batch_accuracy = train_batch_accuracy(
                classifier, train_images_batch, train_labels_batch
            )
            current_validation_accuracy = val_accuracy(
                classifier, val_images, val_labels
            )

            description = (
                f"Step {i};" +
                f"Train Loss => {current_training_loss:0.4};" +
                f"Train Batch Accuracy => {currect_batch_accuracy:0.4};" +
                f"Val Accuracy => {current_validation_accuracy:0.4};" +
                f"Val loss => {current_validation_loss:0.4f}")

            bar.set_description(description)
            bar.refresh()
            bar.refresh()

        # if none of the losses in validation_loss are less than the
        # minimum_val_loss
        if (not current_validation_loss < minimum_val_loss and
                i - minimum_val_step > learning_patience):
            break

    checkpoint.read(
        "artifacts/checkpoints/classify_numbers"
    ).assert_consumed()

    print(f"Final Training Loss => {current_training_loss.numpy():0.4f}")
    print(f"Stop Iteration => {i}")

    final_test_accuracy = test_accuracy(classifier, test_images, test_labels)
    print(f"Test Accuracy => {final_test_accuracy:0.4f}")
