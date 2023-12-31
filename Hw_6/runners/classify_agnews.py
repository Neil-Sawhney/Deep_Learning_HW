from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import yaml
from datasets import load_dataset

from helpers.adam import Adam
from modules.embed_classifier import EmbedClassifier


def train_batch_accuracy(classifier, train_text_batch, train_labels_batch):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                classifier(train_text_batch).numpy().argmax(axis=1),
                train_labels_batch.numpy().reshape(-1),
            ),
            tf.float32,
        )
    )


def val_accuracy(classifier, val_text, val_labels):
    val_accuracy = 0
    val_text = tf.convert_to_tensor(val_text)
    for i in range(0, val_text.shape[0], val_text.shape[0] // 100):
        batch_indices = tf.range(i, i + val_text.shape[0] // 100)
        val_batch_text = tf.gather(val_text, batch_indices)
        val_batch_labels = tf.gather(val_labels, batch_indices)
        if i == 0:
            val_accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        classifier(val_batch_text).numpy().argmax(axis=1),
                        val_batch_labels.numpy().reshape(-1),
                    ),
                    tf.float32,
                )
            ) / (val_text.shape[0] // 100)
        else:
            val_accuracy += tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        classifier(val_batch_text).numpy().argmax(axis=1),
                        val_batch_labels.numpy().reshape(-1),
                    ),
                    tf.float32,
                )
            ) / (val_text.shape[0] // 100)
    return val_accuracy.numpy()


def val_loss(
    classifier,
    val_text,
    val_labels,
    checkpoint_manager,
    minimum_val_loss,
    minimum_val_step,
    current_step,
):
    val_text = tf.convert_to_tensor(val_text)
    validation_loss = 0

    for i in range(0, val_text.shape[0], val_text.shape[0] // 100):
        batch_indices = tf.range(i, i + val_text.shape[0] // 100)
        val_batch_text = tf.gather(val_text, batch_indices)
        val_batch_labels = tf.gather(val_labels, batch_indices)
        validation_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(val_batch_labels), logits=classifier(val_batch_text)
            )
        )

        # average the validation loss over the batches
        if i == 0:
            validation_loss = validation_loss / (val_text.shape[0] // 100)
        else:
            validation_loss += validation_loss / (val_text.shape[0] // 100)

    if validation_loss < minimum_val_loss:
        minimum_val_loss = validation_loss
        minimum_val_step = current_step
        checkpoint_manager.save()

    return validation_loss, minimum_val_loss, minimum_val_step


def val_check(
    classifier,
    val_text,
    val_labels,
    checkpoint_manager,
    minimum_val_loss,
    minimum_val_step_num,
    current_step,
    val_check_rate,
    y_val_loss,
    y_val_accuracy,
    used_patience,
    current_val_loss,
    current_validation_accuracy,
    x_val_iterations,
):
    if current_step % val_check_rate == (val_check_rate - 1):
        (
            current_val_loss,
            minimum_val_loss,
            minimum_val_step_num,
        ) = val_loss(
            classifier,
            val_text,
            val_labels,
            checkpoint_manager,
            minimum_val_loss,
            minimum_val_step_num,
            current_step,
        )
        x_val_iterations = np.append(x_val_iterations, current_step)

        y_val_loss = np.append(y_val_loss, current_val_loss)

        current_validation_accuracy = val_accuracy(classifier, val_text, val_labels)

        y_val_accuracy = np.append(y_val_accuracy, current_validation_accuracy)
        used_patience = current_step - minimum_val_step_num

    return (
        used_patience,
        minimum_val_loss,
        minimum_val_step_num,
        current_validation_accuracy,
        current_val_loss,
        y_val_loss,
        y_val_accuracy,
        x_val_iterations,
    )


def test_accuracy(classifier, test_images, test_labels):
    test_accuracy = 0
    test_images = tf.convert_to_tensor(test_images)
    for i in range(0, test_images.shape[0], test_images.shape[0] // 100):
        batch_indices = tf.range(i, i + test_images.shape[0] // 100)
        test_batch_images = tf.gather(test_images, batch_indices)
        test_batch_labels = tf.gather(test_labels, batch_indices)
        if i == 0:
            test_accuracy = (
                tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            classifier(test_batch_images).numpy().argmax(axis=1),
                            test_batch_labels.numpy().reshape(-1),
                        ),
                        tf.float32,
                    )
                )
                / 100
            )
        else:
            test_accuracy += (
                tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            classifier(test_batch_images).numpy().argmax(axis=1),
                            test_batch_labels.numpy().reshape(-1),
                        ),
                        tf.float32,
                    )
                )
                / 100
            )
    return test_accuracy.numpy()


def train(config_path: Path, use_last_checkpoint: bool):
    if config_path is None:
        config_path = Path("configs/classify_agnews_config.yaml")

    config = yaml.safe_load(config_path.read_text())
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    dropout_prob = config["learning"]["dropout_prob"]
    batch_size = config["learning"]["batch_size"]
    learning_patience = config["learning"]["learning_patience"]
    learning_rates = config["learning"]["learning_rates"]
    num_embeddings = config["learning"]["num_embeddings"]
    embedding_depth = config["learning"]["embedding_depth"]
    val_check_rate = config["learning"]["val_check_rate"]

    refresh_rate = config["display"]["refresh_rate"]

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    num_word_to_tokenize = config["data"]["num_words_to_tokenize"]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    dataset = load_dataset("ag_news")
    train_and_val_labels = dataset["train"]["label"]
    train_and_val_text = dataset["train"]["text"]

    # use 10,000 training samples for validation
    train_labels = train_and_val_labels[:-10000]
    train_text = tf.convert_to_tensor(train_and_val_text[:-10000])
    val_labels = train_and_val_labels[-10000:]
    val_text = train_and_val_text[-10000:]

    minimum_val_step_num = 0
    current_val_loss = 0
    used_patience = 0
    current_val_loss = -1
    current_validation_accuracy = -1

    num_classes = 4
    embed_classifier = EmbedClassifier(
        num_embeddings,
        embedding_depth,
        num_word_to_tokenize,
        dropout_prob,
        num_hidden_layers,
        hidden_layer_width,
        num_classes,
    )

    # Used For Plotting
    y_train_batch_accuracy = np.array([])
    y_train_batch_loss = np.array([])
    y_val_accuracy = np.array([])
    y_val_loss = np.array([])
    x_train_loss_iterations = np.array([])
    x_train_accuracy_iterations = np.array([])
    x_val_iterations = np.array([])

    learning_rate_change_steps = np.array([])

    # Index of the current learning rate, used to change the learning rate
    # when the validation loss stops improving
    learning_rate_index = 0
    adam = Adam(
        learning_rates[learning_rate_index],
        weight_decay=weight_decay,
    )

    checkpoint = tf.train.Checkpoint(embed_classifier)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "temp/checkpoints/classify_agnews", max_to_keep=1
    )
    if use_last_checkpoint:
        print("\n\nRestoring from last checkpoint")
        checkpoint_manager.restore_or_initialize()

    overall_log = tqdm.tqdm(total=0, position=1, bar_format="{desc}")
    train_log = tqdm.tqdm(total=0, position=2, bar_format="{desc}")
    val_log = tqdm.tqdm(total=0, position=3, bar_format="{desc}")
    bar = tqdm.trange(num_iters, position=4)

    num_of_parameters = tf.math.add_n(
        [tf.math.reduce_prod(var.shape) for var in embed_classifier.trainable_variables]
    )
    print(f"\nNumber of Parameters => {num_of_parameters}")

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=train_text.shape[0], dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            train_text_batch = tf.gather(train_text, batch_indices)
            train_labels_batch = tf.gather(train_labels, batch_indices)

            current_train_batch_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(train_labels_batch),
                    logits=embed_classifier(train_text_batch),
                )
            )

        # Print initial train batch loss
        if i == 0:
            print("\n\n\n\n")
            print(f"Initial Training Loss => {current_train_batch_loss:0.4f}")
            _, minimum_val_loss, minimum_val_step_num = val_loss(
                embed_classifier,
                val_text,
                val_labels,
                checkpoint_manager,
                np.inf,
                i,
                i,
            )
            current_validation_accuracy = val_accuracy(
                embed_classifier, val_text, val_labels
            )
            current_val_loss = minimum_val_loss

        grads = tape.gradient(
            current_train_batch_loss, embed_classifier.trainable_variables
        )

        adam.apply_gradients(zip(grads, embed_classifier.trainable_variables))

        (
            used_patience,
            minimum_val_loss,
            minimum_val_step_num,
            current_validation_accuracy,
            current_val_loss,
            y_val_loss,
            y_val_accuracy,
            x_val_iterations,
        ) = val_check(
            embed_classifier,
            val_text,
            val_labels,
            checkpoint_manager,
            minimum_val_loss,
            minimum_val_step_num,
            i,
            val_check_rate,
            y_val_loss,
            y_val_accuracy,
            used_patience,
            current_val_loss,
            current_validation_accuracy,
            x_val_iterations,
        )

        current_train_batch_loss = current_train_batch_loss.numpy()
        y_train_batch_loss = np.append(y_train_batch_loss, current_train_batch_loss)
        x_train_loss_iterations = np.append(x_train_loss_iterations, i)

        if i % refresh_rate == (refresh_rate - 1):
            current_batch_accuracy = train_batch_accuracy(
                embed_classifier, train_text_batch, train_labels_batch
            )
            x_train_accuracy_iterations = np.append(x_train_accuracy_iterations, i)
            y_train_batch_accuracy = np.append(
                y_train_batch_accuracy, current_batch_accuracy
            )

            learning_rates_left = len(learning_rates) - learning_rate_index
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
                f"Val Loss => {current_val_loss:0.4f}    "
                + f"Val Accuracy => {current_validation_accuracy:0.4f}    "
            )
            val_log.set_description_str(val_description)
            val_log.update(refresh_rate)

            bar_description = f"Step => {i}"
            bar.set_description(bar_description)
            bar.refresh()

            # if the validation loss has not improved for learning_patience
            if (
                current_val_loss > minimum_val_loss
                and i - minimum_val_step_num > learning_patience
            ):
                if learning_rate_index == (len(learning_rates) - 1):
                    break
                learning_rate_index += 1
                adam.learning_rate = learning_rates[learning_rate_index]
                learning_rate_change_steps = np.append(learning_rate_change_steps, i)
                minimum_val_step_num = i
                checkpoint_manager.restore_or_initialize()

    checkpoint_manager.restore_or_initialize()
    current_validation_accuracy = val_accuracy(embed_classifier, val_text, val_labels)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(
        x_train_accuracy_iterations, y_train_batch_accuracy, label="Train Accuracy"
    )
    ax[0].plot(x_val_iterations, y_val_accuracy, label="Val Accuracy")
    # plot vertical line on learning rate change
    for learning_rate_change_step in learning_rate_change_steps:
        ax[0].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].semilogy(
        x_train_loss_iterations, y_train_batch_loss, label="Train Batch Loss"
    )
    ax[1].semilogy(x_val_iterations, y_val_loss, label="Val Loss")
    for learning_rate_change_step in learning_rate_change_steps:
        ax[1].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    print("\n\n\n\n")
    print(f"Final Training Loss => {current_train_batch_loss:0.4f}")
    print(f"Stop Iteration => {i}")

    fig.suptitle(
        "Classify AGNews: Final Val Accuracy = " + f"{current_validation_accuracy:0.4f}"
    )

    # if the file already exists add a number to the end of the file name
    # to avoid overwriting
    file_index = 0
    while Path(f"artifacts/agnews/classify_agnews_img_{file_index}.png").exists():
        file_index += 1
    fig.savefig(f"artifacts/agnews/classify_agnews_img_{file_index}.png")

    # Save the config file as a yaml under the same name as the image
    config_path = Path(f"artifacts/agnews/classify_agnews_img_{file_index}.yaml")
    config_path.write_text(yaml.dump(config))

    # save the model
    checkpoint_manager.save()
    config_path = Path(f"artifacts/agnews/model.yaml")
    config_path.write_text(yaml.dump(config))


def test(checkpoint_path: Path):
    if checkpoint_path is None:
        checkpoint_path = Path("temp/checkpoints/classify_agnews")

    if not checkpoint_path.exists():
        print("Checkpoint does not exist, run the train script first")
        return

    config_path = Path("artifacts/agnews/model.yaml")

    config = yaml.safe_load(config_path.read_text())
    dropout_prob = config["learning"]["dropout_prob"]
    num_embeddings = config["learning"]["num_embeddings"]
    embedding_depth = config["learning"]["embedding_depth"]

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    num_word_to_tokenize = config["data"]["num_words_to_tokenize"]

    num_classes = 4
    embed_classifier = EmbedClassifier(
        num_embeddings,
        embedding_depth,
        num_word_to_tokenize,
        dropout_prob,
        num_hidden_layers,
        hidden_layer_width,
        num_classes,
    )

    checkpoint = tf.train.Checkpoint(embed_classifier)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    dataset = load_dataset("ag_news")
    test_labels = dataset["test"]["label"]
    test_text = dataset["test"]["text"]

    test_text = tf.convert_to_tensor(test_text)
    test_labels = tf.convert_to_tensor(test_labels)

    test_accuracy_value = test_accuracy(embed_classifier, test_text, test_labels)

    print(f"Test Accuracy => {test_accuracy_value:0.4f}")
