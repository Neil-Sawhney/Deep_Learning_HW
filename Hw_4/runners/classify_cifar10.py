from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import yaml

from helpers.augment_data import augment_data
from helpers.load_pickle_data import load_pickle_data
from helpers.optimizer import Adam
from modules.classifier import Classifier


def train_batch_accuracy(classifier, train_images_batch, train_labels_batch):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(train_images_batch).numpy().argmax(axis=1),
                train_labels_batch.numpy().reshape(-1),
            ),
            tf.float32,
        )
    )


def val_accuracy(classifier, val_images, val_labels):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(val_images).numpy().argmax(axis=1),
                val_labels.numpy().reshape(-1),
            ),
            tf.float32,
        )
    )


def test_accuracy(classifier, test_images, test_labels):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(test_images).numpy().argmax(axis=1),
                test_labels.numpy().reshape(-1),
            ),
            tf.float32,
        )
    ).numpy()


def val_loss(
    classifier,
    val_images,
    val_labels,
    checkpoint_manager,
    minimum_val_loss,
    minimum_val_step,
    current_step,
):
    validation_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(val_labels), logits=classifier(val_images)
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

    config = yaml.safe_load(config_path.read_text())
    layer_depths = config["cnn"]["layer_depths"]
    kernel_sizes = config["cnn"]["kernel_sizes"]
    pool_every_n_layers = config["cnn"]["pool_every_n_layers"]
    resblock_size = config["cnn"]["resblock_size"]
    pool_size = config["cnn"]["pool_size"]
    num_classes = config["cnn"]["num_classes"]
    num_augmentations = config["cnn"]["num_augmentations"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    dropout_prob = config["learning"]["dropout_prob"]
    batch_size = config["learning"]["batch_size"]
    learning_patience = config["learning"]["learning_patience"]
    dropout_first_n_fc_layers = config["learning"]["dropout_first_n_fc_layers"]
    learning_rates = config["learning"]["learning_rates"]
    refresh_rate = config["display"]["refresh_rate"]
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    train_and_val_labels, train_and_val_images = load_pickle_data(
        "data/cifar-10-batches-py/data_batch_1"
    )
    for i in range(2, 6):
        labels, images = load_pickle_data(f"data/cifar-10-batches-py/data_batch_{i}")
        train_and_val_labels = tf.concat([train_and_val_labels, labels], axis=0)
        train_and_val_images = tf.concat([train_and_val_images, images], axis=0)

    train_labels = train_and_val_labels[:-10000]
    train_images = train_and_val_images[:-10000]
    val_labels = train_and_val_labels[-10000:]
    val_images = train_and_val_images[-10000:]

    test_labels, test_images = load_pickle_data("data/cifar-10-batches-py/test_batch")

    num_samples = train_images.shape[0]
    input_depth = train_images.shape[-1]
    classifier = Classifier(
        input_depth,
        layer_depths,
        kernel_sizes,
        num_classes,
        train_images.shape[1],
        num_hidden_layers,
        hidden_layer_width,
        resblock_size,
        pool_every_n_layers,
        pool_size,
        dropout_first_n_fc_layers,
        dropout_prob,
    )

    minimum_val_loss = float("inf")
    minimum_val_step_num = 0
    current_validation_loss = 0

    # Used For Plotting
    y_train_batch_accuracy = np.array([])
    y_train_batch_loss = np.array([])
    y_val_accuracy = np.array([])
    y_val_loss = np.array([])
    x_loss_iterations = np.array([])
    x_accuracy_iterations = np.array([])

    learning_rate_change_steps = np.array([])

    # Index of the current learning rate, used to change the learning rate
    # when the validation loss stops improving
    learning_rate_index = 0
    adam = Adam(
        learning_rates[learning_rate_index],
        weight_decay=weight_decay,
    )

    checkpoint = tf.train.Checkpoint(classifier)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "temp/checkpoints/classify_numbers", max_to_keep=1
    )
    if use_last_checkpoint:
        print("\n\nRestoring from last checkpoint")
        checkpoint_manager.restore_or_initialize()

    overall_log = tqdm.tqdm(total=0, position=1, bar_format="{desc}")
    train_log = tqdm.tqdm(total=0, position=2, bar_format="{desc}")
    val_log = tqdm.tqdm(total=0, position=3, bar_format="{desc}")
    bar = tqdm.trange(num_iters, position=4)

    num_of_parameters = tf.math.add_n(
        [tf.math.reduce_prod(var.shape) for var in classifier.trainable_variables]
    )
    print(f"\nNumber of Parameters => {num_of_parameters}")

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            train_images_batch = tf.gather(train_images, batch_indices)
            train_labels_batch = tf.gather(train_labels, batch_indices)
            current_train_batch_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(train_labels_batch),
                    logits=classifier(
                        augment_data(train_images_batch, num_augmentations)),
                )
            )

        # Print initial train batch loss
        if i == 0:
            print("\n\n\n\n")
            print(f"Initial Training Loss => {current_train_batch_loss:0.4f}")

        grads = tape.gradient(current_train_batch_loss, classifier.trainable_variables)

        adam.apply_gradients(zip(grads, classifier.trainable_variables))

        current_validation_loss, minimum_val_loss, minimum_val_step_num = val_loss(
            classifier,
            val_images,
            val_labels,
            checkpoint_manager,
            minimum_val_loss,
            minimum_val_step_num,
            i,
        )

        y_val_loss = np.append(y_val_loss, current_validation_loss)
        current_train_batch_loss = current_train_batch_loss.numpy()
        y_train_batch_loss = np.append(y_train_batch_loss, current_train_batch_loss)
        x_loss_iterations = np.append(x_loss_iterations, i)

        if i % refresh_rate == (refresh_rate - 1):
            current_batch_accuracy = train_batch_accuracy(
                classifier, train_images_batch, train_labels_batch
            )
            current_validation_accuracy = val_accuracy(
                classifier, val_images, val_labels
            )

            x_accuracy_iterations = np.append(x_accuracy_iterations, i)
            y_train_batch_accuracy = np.append(
                y_train_batch_accuracy, current_batch_accuracy
            )
            y_val_accuracy = np.append(y_val_accuracy, current_validation_accuracy)

            learning_rates_left = len(learning_rates) - learning_rate_index
            used_patience = i - minimum_val_step_num
            patience_left = learning_patience - used_patience
            overall_description = (
                f"Minimum Val Loss => {minimum_val_loss:0.4f}    "
                + f"Learning Rates Left => {learning_rates_left}    "
                + f"Patience Left => {patience_left}    "
            )
            overall_log.set_description_str(overall_description)
            overall_log.refresh()

            train_description = (
                f"Train Batch Loss => {current_train_batch_loss:0.4f}    "
                + f"Train Accuracy => {current_batch_accuracy:0.4f}    "
            )
            train_log.set_description_str(train_description)
            train_log.update(refresh_rate)

            val_description = (
                f"Val Loss => {current_validation_loss:0.4f}    "
                + f"Val Accuracy => {current_validation_accuracy:0.4f}    "
            )
            val_log.set_description_str(val_description)
            val_log.update(refresh_rate)

            bar_description = f"Step => {i}"
            bar.set_description(bar_description)
            bar.refresh()

            # if the validation loss has not improved for learning_patience
            if (
                current_validation_loss > minimum_val_loss
                and i - minimum_val_step_num > learning_patience
            ):
                if learning_rate_index == (len(learning_rates) - 1):
                    break
                learning_rate_index += 1
                adam.learning_rate = learning_rates[learning_rate_index]
                learning_rate_change_steps = np.append(learning_rate_change_steps, i)
                checkpoint_manager.restore_or_initialize()

    checkpoint_manager.restore_or_initialize()

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(x_accuracy_iterations, y_train_batch_accuracy, label="Train Accuracy")
    ax[0].plot(x_accuracy_iterations, y_val_accuracy, label="Val Accuracy")
    # plot vertical line on learning rate change
    for learning_rate_change_step in learning_rate_change_steps:
        ax[0].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].semilogy(x_loss_iterations, y_train_batch_loss, label="Train Loss")
    ax[1].semilogy(x_loss_iterations, y_val_loss, label="Val Loss")
    for learning_rate_change_step in learning_rate_change_steps:
        ax[1].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    print("\n\n\n\n")
    print(f"Final Training Loss => {current_train_batch_loss:0.4f}")
    print(f"Stop Iteration => {i}")

    final_test_accuracy = test_accuracy(classifier, test_images, test_labels)
    print(f"Test Accuracy => {final_test_accuracy:0.4f}")
    fig.suptitle("Classify Cifar10: Test Accuracy = " + str(final_test_accuracy))

    # if the file already exists add a number to the end of the file name
    # to avoid overwriting
    file_index = 0
    while Path(f"artifacts/classify_cifar10_img_{file_index}.png").exists():
        file_index += 1
    fig.savefig(f"artifacts/classify_cifar10_img_{file_index}.png")
    with open(f"artifacts/classify_cifar10_log_{file_index}.txt", "w") as file:
        file.write(f"Config => {config}\n")
        file.write(f"Test Accuracy => {final_test_accuracy:0.4f}\n")
        file.write(f"Stop Iteration => {file_index}\n")
