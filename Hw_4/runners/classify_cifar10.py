from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange

from helpers.load_pickle_data import load_pickle_data
from helpers.optimizer import Adam
from modules.classifier import Classifier


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
             checkpoint_manager,
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
        checkpoint_manager.save()

    return validation_loss, minimum_val_loss, minimum_val_step


def run(config_path: Path, use_last_checkpoint: bool):
    if config_path is None:
        config_path = Path("configs/classify_cifar10_config.yaml")

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    train_and_val_labels, train_and_val_images = load_pickle_data(
        "data/cifar-10-batches-py/data_batch_1")
    for i in range(2, 6):
        labels, images = load_pickle_data(
            f"data/cifar-10-batches-py/data_batch_{i}")
        train_and_val_labels = tf.concat([train_and_val_labels, labels],
                                         axis=0)
        train_and_val_images = tf.concat([train_and_val_images, images],
                                         axis=0)

    train_labels = train_and_val_labels[:-10000]
    train_images = train_and_val_images[:-10000]
    val_labels = train_and_val_labels[-10000:]
    val_images = train_and_val_images[-10000:]

    test_labels, test_images = load_pickle_data(
        "data/cifar-10-batches-py/test_batch")

    config = yaml.safe_load(config_path.read_text())
    layer_depths = config["cnn"]["layer_depths"]
    kernel_sizes = config["cnn"]["kernel_sizes"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    dropout_prob = config["learning"]["dropout_prob"]
    batch_size = config["learning"]["batch_size"]
    learning_patience = config["learning"]["learning_patience"]
    dropout_first_n_fc_layers = config["learning"]["dropout_first_n_fc_layers"]
    learning_rates = config["learning"]["learning_rates"]
    refresh_rate = config["display"]["refresh_rate"]
    pool_every_n_layers = config["cnn"]["pool_every_n_layers"]
    pool_size = config["cnn"]["pool_size"]
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]
    num_samples = train_images.shape[0]
    input_depth = train_images.shape[-1]
    num_classes = 10
    classifier = Classifier(input_depth,
                            layer_depths,
                            kernel_sizes,
                            num_classes,
                            train_images.shape[1],
                            num_hidden_layers,
                            hidden_layer_width,
                            pool_every_n_layers,
                            pool_size,
                            dropout_first_n_fc_layers,
                            dropout_prob)

    minimum_val_loss = float("inf")
    minimum_val_step_num = 0
    current_validation_loss = 0

    # Used For Plotting
    y_train_batch_accuracy = np.array([])
    y_val_accuracy = np.array([])
    x_iterations = np.array([])

    checkpoint = tf.train.Checkpoint(classifier)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "artifacts/checkpoints/classify_numbers", max_to_keep=1
    )
    if use_last_checkpoint:
        checkpoint_manager.restore_or_initialize()

    # Index of the current learning rate, used to change the learning rate
    # when the validation loss stops improving
    learning_rate_index = 0
    adam = Adam(learning_rates[learning_rate_index],
                weight_decay=weight_decay,)

    bar = trange(num_iters)
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
            zip(grads, classifier.trainable_variables)
        )

        # If no improvement in validation loss for learning_patience
        # iterations, stop training
        current_validation_loss, minimum_val_loss, minimum_val_step_num = \
            val_loss(classifier,
                     val_images,
                     val_labels,
                     checkpoint_manager,
                     minimum_val_loss,
                     minimum_val_step_num,
                     i)

        if i % refresh_rate == (refresh_rate - 1):
            current_training_loss = current_training_loss.numpy()
            current_batch_accuracy = train_batch_accuracy(
                classifier, train_images_batch, train_labels_batch
            )
            current_validation_accuracy = val_accuracy(
                classifier, val_images, val_labels
            )

            y_train_batch_accuracy = np.append(y_train_batch_accuracy,
                                               current_batch_accuracy)
            y_val_accuracy = np.append(y_val_accuracy,
                                       current_validation_accuracy)
            x_iterations = np.append(x_iterations, i)

            description = (
                f"Minimum Val Loss => {minimum_val_loss:0.4f};\n" +
                f"Step {i};" +
                f"Train Loss => {current_training_loss:0.4};" +
                f"Train Batch Accuracy => {current_batch_accuracy:0.4};" +
                f"Val Accuracy => {current_validation_accuracy:0.4};" +
                f"Val loss => {current_validation_loss:0.4f}")

            bar.set_description(description)
            bar.refresh()

            # if the validation loss has not improved for learning_patience
            if (current_validation_loss > minimum_val_loss and
                    i - minimum_val_step_num > learning_patience):
                if (learning_rate_index == (len(learning_rates) - 1)):
                    break
                learning_rate_index += 1
                adam.learning_rate = learning_rates[learning_rate_index]

    checkpoint_manager.restore_or_initialize()

    print(f"Final Training Loss => {current_training_loss:0.4f}")
    print(f"Stop Iteration => {i}")

    final_test_accuracy = test_accuracy(classifier, test_images, test_labels)
    print(f"Test Accuracy => {final_test_accuracy:0.4f}")

    fig, ax = plt.subplots()
    ax.plot(x_iterations, y_train_batch_accuracy, label="Train Accuracy")
    ax.plot(x_iterations, y_val_accuracy, label="Val Accuracy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Accuracy vs Iteration: Test Accuracy = "
                 + str(format(final_test_accuracy, '.4f')))
    # if the file already exists add a number to the end of the file name
    # to avoid overwriting
    i = 0
    while Path(f"artifacts/images/classify_cifar10_accuracy_{i}.png").exists():
        i += 1
    fig.savefig(f"artifacts/images/classify_cifar10_accuracy_{i}.png")
