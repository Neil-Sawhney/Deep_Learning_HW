from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import yaml
from datasets import load_dataset

from helpers.adam import Adam
from helpers.tokenizer import Tokenizer
from modules.transformer_decoder import TransformerDecoder


def train_batch_accuracy(logits, labels):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                logits.numpy().argmax(axis=2).reshape(-1),
                labels.numpy().reshape(-1),
            ),
            tf.float32,
        )
    )


def train(config_path: Path, use_last_checkpoint: bool):
    if config_path is None:
        config_path = Path("configs/predict_who_bites_who.yaml")

    # HYPERPARAMETERS
    config = yaml.safe_load(config_path.read_text())
    refresh_rate = config["display"]["refresh_rate"]
    batch_size = config["learning"]["batch_size"]
    learning_patience = config["learning"]["learning_patience"]
    learning_rates = config["learning"]["learning_rates"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    context_length = config["data"]["context_length"]
    min_vocab_size = config["data"]["min_vocab_size"]
    num_heads = config["transformer"]["num_heads"]
    model_dim = config["transformer"]["model_dim"]
    ffn_dim = config["transformer"]["ffn_dim"]
    num_blocks = config["transformer"]["num_blocks"]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    # TODO: padding mask and causal mask, using the embedder
    with open("data/who_bites_who.txt", "r", encoding="utf-8") as f:
        one_big_slab_of_text = f.read()

    tokenizer = Tokenizer(context_length, False)
    tokenized_text = tokenizer(one_big_slab_of_text)

    # add the start token
    targets = tokenized_text
    hashed_targets = tf.strings.to_hash_bucket_fast(targets, min_vocab_size)

    tokenized_text = tf.concat(
        [
            tf.fill([tokenized_text.shape[0], 1], b"<START>"),
            tokenized_text,
        ],
        axis=1,
    )
    tokenized_text = tokenized_text[:, :-1]

    transformer_decoder = TransformerDecoder(
        min_vocab_size,
        context_length,
        num_heads,
        model_dim,
        ffn_dim,
        num_blocks,
    )

    used_patience = 0
    minimum_train_loss = np.inf
    minimum_loss_step_num = 0

    # Used For Plotting
    y_train_accuracy = np.array([])
    y_train_batch_loss = np.array([])
    x_train_loss_iterations = np.array([])
    x_train_accuracy_iterations = np.array([])

    learning_rate_change_steps = np.array([])

    # Index of the current learning rate, used to change the learning rate
    # when the training loss stops improving
    learning_rate_index = 0
    adam = Adam(
        learning_rates[learning_rate_index],
        weight_decay=weight_decay,
    )

    checkpoint = tf.train.Checkpoint(transformer_decoder)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "temp/checkpoints/predict_who_bites_who", max_to_keep=1
    )
    if use_last_checkpoint:
        print("\n\nRestoring from last checkpoint")
        checkpoint_manager.restore_or_initialize()

    overall_log = tqdm.tqdm(total=0, position=1, bar_format="{desc}")
    train_log = tqdm.tqdm(total=0, position=2, bar_format="{desc}")
    bar = tqdm.trange(num_iters, position=3)

    num_of_parameters = tf.math.add_n(
        [
            tf.math.reduce_prod(var.shape)
            for var in transformer_decoder.trainable_variables
        ]
    )
    print(f"\nNumber of Parameters => {num_of_parameters}")

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=tokenized_text.shape[0], dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            input_tokens_batch = tf.gather(tokenized_text, batch_indices)
            targets_batch = tf.gather(hashed_targets, batch_indices)

            labels = targets_batch
            logits = transformer_decoder(input_tokens_batch)
            current_train_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits,
                )
            )

        # Print initial train batch loss
        if i == 0:
            print("\n\n\n\n")
            print(f"Initial Training Loss => {current_train_loss:0.4f}")

        grads = tape.gradient(
            current_train_loss, transformer_decoder.trainable_variables
        )

        adam.apply_gradients(zip(grads, transformer_decoder.trainable_variables))

        current_train_loss = current_train_loss.numpy()

        if current_train_loss < minimum_train_loss:
            minimum_train_loss = current_train_loss
            minimum_loss_step_num = i
            checkpoint_manager.save()
        used_patience = i - minimum_loss_step_num

        y_train_batch_loss = np.append(y_train_batch_loss, current_train_loss)
        x_train_loss_iterations = np.append(x_train_loss_iterations, i)

        if i % refresh_rate == (refresh_rate - 1):
            current_batch_accuracy = train_batch_accuracy(logits, labels)
            x_train_accuracy_iterations = np.append(x_train_accuracy_iterations, i)
            y_train_accuracy = np.append(y_train_accuracy, current_batch_accuracy)

            learning_rates_left = len(learning_rates) - learning_rate_index
            patience_left = learning_patience - used_patience
            overall_description = (
                f"Minimum Train Loss => {minimum_train_loss:0.4f}    "
                + f"Learning Rates Left => {learning_rates_left}    "
                + f"Patience Left => {patience_left}    "
            )
            overall_log.set_description_str(overall_description)
            overall_log.refresh()

            train_description = (
                f"Train Batch Loss => {current_train_loss:0.4f}    "
                + f"Train Accuracy => {current_batch_accuracy:0.4f}    "
            )
            train_log.set_description_str(train_description)
            train_log.update(refresh_rate)

            bar_description = f"Step => {i}"
            bar.set_description(bar_description)
            bar.refresh()

            # if the training loss has not improved for learning_patience
            if (
                current_train_loss > minimum_train_loss
                and i - minimum_loss_step_num > learning_patience
            ):
                if learning_rate_index == (len(learning_rates) - 1):
                    break
                learning_rate_index += 1
                adam.learning_rate = learning_rates[learning_rate_index]
                learning_rate_change_steps = np.append(learning_rate_change_steps, i)
                minimum_loss_step_num = i
                checkpoint_manager.restore_or_initialize()

    checkpoint_manager.restore_or_initialize()
    batch_indices = rng.uniform(
        shape=[batch_size], maxval=tokenized_text.shape[0], dtype=tf.int32
    )
    input_tokens_batch = tf.gather(tokenized_text, batch_indices)
    targets_batch = tf.gather(hashed_targets, batch_indices)

    labels = targets_batch
    logits = transformer_decoder(input_tokens_batch)
    final_train_accuracy = train_batch_accuracy(logits, labels)

    fig, ax = plt.subplots(2, 1)

    ax[0].semilogy(x_train_loss_iterations, y_train_batch_loss, label="Training Loss")
    for learning_rate_change_step in learning_rate_change_steps:
        ax[0].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Loss")

    ax[1].semilogy(
        x_train_accuracy_iterations, y_train_accuracy, label="Training Accuracy"
    )
    for learning_rate_change_step in learning_rate_change_steps:
        ax[1].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Accuracy")

    print("\n\n\n\n")
    # FIXME:why is it higher at the end
    print(f"Final Training Loss => {current_train_loss:0.4f}")
    print(f"Stop Iteration => {i}")

    fig.suptitle(
        "Predict Who Bites Who: Final Train Accuracy = "
        + f"{final_train_accuracy:0.4f}"
    )

    # if the file already exists add a number to the end of the file name
    # to avoid overwriting
    file_index = 0
    while Path(
        f"artifacts/who_bites_who/predict_who_bites_who_img_{file_index}.png"
    ).exists():
        file_index += 1
    fig.savefig(f"artifacts/who_bites_who/predict_who_bites_who_img_{file_index}.png")

    # Save the config file as a yaml under the same name as the image
    config_path = Path(
        f"artifacts/who_bites_who/predict_who_bites_who_img_{file_index}.yaml"
    )
    config_path.write_text(yaml.dump(config))

    # save the model
    checkpoint_manager.directory = "artifacts/who_bites_who/model"
    checkpoint_manager.save()
    config_path = Path(f"artifacts/who_bites_who/model.yaml")
    config_path.write_text(yaml.dump(config))


def test(model_path: Path):
    if model_path is None:
        model_path = Path("artifacts/who_bites_who/model")

    if not model_path.exists():
        print("Model does not exist, run the train script first")
        return

    config_path = Path("artifacts/agnews/model/model.yaml")

    config = yaml.safe_load(config_path.read_text())
    dropout_prob = config["learning"]["dropout_prob"]
    num_embeddings = config["learning"]["num_embeddings"]
    embedding_depth = config["learning"]["embedding_depth"]

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    num_word_to_tokenize = config["data"]["num_words_to_tokenize"]

    # TODO: num_classes is now the number of tokens add that

    # CLEANUP: remove
    # num_classes = 4
    # embed_classifier = EmbedClassifier(
    #     num_embeddings,
    #     embedding_depth,
    #     num_word_to_tokenize,
    #     dropout_prob,
    #     num_hidden_layers,
    #     hidden_layer_width,
    #     num_classes,
    # )

    checkpoint = tf.train.Checkpoint(TransformerDecoder)
    checkpoint.restore(tf.train.latest_checkpoint(model_path))

    dataset = load_dataset("ag_news")
    test_labels = dataset["test"]["label"]
    test_text = dataset["test"]["text"]

    test_text = tf.convert_to_tensor(test_text)
    test_labels = tf.convert_to_tensor(test_labels)

    test_accuracy_value = test_accuracy(TransformerDecoder, test_text, test_labels)

    print(f"Test Accuracy => {test_accuracy_value:0.4f}")

    file = open("artifacts/agnews/classify_agnews_test_accuracy.txt", "w")
    file.write(f"{test_accuracy_value:0.4f}")
    file.close()
