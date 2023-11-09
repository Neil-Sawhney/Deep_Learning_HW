from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import yaml
from datasets import load_dataset

from helpers.adam import Adam
from helpers.embedder import Embedder
from helpers.tokenizer import Tokenizer
from modules.transformer_decoder import TransformerDecoder


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


def train(config_path: Path, use_last_checkpoint: bool):
    if config_path is None:
        config_path = Path("configs/predict_who_bites_who.yaml")

    # HYPERPARAMETERS
    config = yaml.safe_load(config_path.read_text())
    refresh_rate = config["display"]["refresh_rate"]
    val_check_rate = config["learning"]["val_check_rate"]
    batch_size = config["learning"]["batch_size"]
    learning_patience = config["learning"]["learning_patience"]
    learning_rates = config["learning"]["learning_rates"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    num_embeddings = config["learning"]["num_embeddings"]
    embedding_depth = config["learning"]["embedding_depth"]
    context_length = config["data"]["context_length"]
    num_heads = config["transformer"]["num_heads"]
    model_dim = config["transformer"]["model_dim"]
    ffn_dim = config["transformer"]["ffn_dim"]
    num_blocks = config["transformer"]["num_blocks"]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    # TODO: padding mask and causal mask, using the embedder, positional encoding
    with open("data/who_bites_who.txt", "r", encoding="utf-8") as f:
        one_big_slab_of_text = f.read()

    tokenizer = Tokenizer(context_length, False)
    tokenized_text = tokenizer(one_big_slab_of_text)

    embedder = Embedder(num_embeddings, embedding_depth)
    # TODO: add positional encoding later
    input_embedding = embedder(tokenized_text)

    transformer_decoder = TransformerDecoder(
        num_embeddings,
        embedding_depth,
        context_length,
        num_heads,
        model_dim,
        ffn_dim,
        num_blocks,
    )

    # # CLEANUP: remove
    # # dataset = load_dataset("ag_news")
    # # train_and_val_labels = dataset["train"]["label"]
    # # train_and_val_text = dataset["train"]["text"]
    # #
    # # train_labels = train_and_val_labels[:-10000]
    # # train_text = tf.convert_to_tensor(train_and_val_text[:-10000])
    # # val_labels = train_and_val_labels[-10000:]
    # # val_text = train_and_val_text[-10000:]

    used_patience = 0

    # # CLEANUP: remove
    # # num_classes = 4
    # # embed_classifier = EmbedClassifier(
    # #     num_embeddings,
    # #     embedding_depth,
    # #     num_word_to_tokenize,
    # #     dropout_prob,
    # #     num_hidden_layers,
    # #     hidden_layer_width,
    # #     num_classes,
    # # )

    # Used For Plotting
    y_train_batch_accuracy = np.array([])
    y_train_batch_loss = np.array([])
    x_train_loss_iterations = np.array([])
    x_train_accuracy_iterations = np.array([])

    learning_rate_index = 0  # Index of the current learning rate, used to change the learning rate # when the train batch loss stops improving
    learning_rate_change_steps = np.array([])
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
            shape=[batch_size], maxval=train_text.shape[0], dtype=tf.int32
        )
        # TODO: split each entry of size context length in the batch dimension into random sequences of all lengths <= (context_length - 1). The target is the next token in the sequence.
        window_start_index = tf.random.uniform(
            shape=(), maxval=text.shape[1] - context_length, dtype=tf.int32
        )

    #     with tf.GradientTape() as tape:
    #         input_embedding_batch = tf.gather(train_text, batch_indices)
    #         targets_batch = tf.gather(train_labels, batch_indices)

    #         current_train_batch_loss = tf.reduce_mean(
    #             tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                 labels=tf.squeeze(target_batch),
    #                 logits=TransformerDecoder(text_batch),
    #             )
    #         )

    #     # Print initial train batch loss
    #     if i == 0:
    #         print("\n\n\n\n")
    #         print(f"Initial Training Loss => {current_train_batch_loss:0.4f}")
    #         _, minimum_val_loss, minimum_val_step_num = val_loss(
    #             TransformerDecoder,
    #             val_text,
    #             val_labels,
    #             checkpoint_manager,
    #             np.inf,
    #             i,
    #             i,
    #         )
    #         current_validation_accuracy = val_accuracy(
    #             TransformerDecoder, val_text, val_labels
    #         )
    #         current_val_loss = minimum_val_loss

    #     grads = tape.gradient(
    #         current_train_batch_loss, TransformerDecoder.trainable_variables
    #     )

    #     adam.apply_gradients(zip(grads, TransformerDecoder.trainable_variables))

    #     (
    #         used_patience,
    #         minimum_val_loss,
    #         minimum_val_step_num,
    #         current_validation_accuracy,
    #         current_val_loss,
    #         y_val_loss,
    #         y_val_accuracy,
    #         x_val_iterations,
    #     ) = val_check(
    #         TransformerDecoder,
    #         val_text,
    #         val_labels,
    #         checkpoint_manager,
    #         minimum_val_loss,
    #         minimum_val_step_num,
    #         i,
    #         val_check_rate,
    #         y_val_loss,
    #         y_val_accuracy,
    #         used_patience,
    #         current_val_loss,
    #         current_validation_accuracy,
    #         x_val_iterations,
    #     )

    #     current_train_batch_loss = current_train_batch_loss.numpy()
    #     y_train_batch_loss = np.append(y_train_batch_loss, current_train_batch_loss)
    #     x_train_loss_iterations = np.append(x_train_loss_iterations, i)

    #     if i % refresh_rate == (refresh_rate - 1):
    #         current_batch_accuracy = train_batch_accuracy(
    #             TransformerDecoder, text_batch, target_batch
    #         )
    #         x_train_accuracy_iterations = np.append(x_train_accuracy_iterations, i)
    #         y_train_batch_accuracy = np.append(
    #             y_train_batch_accuracy, current_batch_accuracy
    #         )

    #         learning_rates_left = len(learning_rates) - learning_rate_index
    #         patience_left = learning_patience - used_patience
    #         overall_description = (
    #             f"Minimum Val Loss => {minimum_val_loss:0.4f}    "
    #             + f"Learning Rates Left => {learning_rates_left}    "
    #             + f"Patience Left => {patience_left}    "
    #         )
    #         overall_log.set_description_str(overall_description)
    #         overall_log.refresh()

    #         train_description = (
    #             f"Train Batch Loss => {current_train_batch_loss:0.4f}    "
    #             + f"Train Accuracy => {current_batch_accuracy:0.4f}    "
    #         )
    #         train_log.set_description_str(train_description)
    #         train_log.update(refresh_rate)

    #         val_description = (
    #             f"Val Loss => {current_val_loss:0.4f}    "
    #             + f"Val Accuracy => {current_validation_accuracy:0.4f}    "
    #         )
    #         val_log.set_description_str(val_description)
    #         val_log.update(refresh_rate)

    #         bar_description = f"Step => {i}"
    #         bar.set_description(bar_description)
    #         bar.refresh()

    #         # if the validation loss has not improved for learning_patience
    #         if (
    #             current_val_loss > minimum_val_loss
    #             and i - minimum_val_step_num > learning_patience
    #         ):
    #             if learning_rate_index == (len(learning_rates) - 1):
    #                 break
    #             learning_rate_index += 1
    #             adam.learning_rate = learning_rates[learning_rate_index]
    #             learning_rate_change_steps = np.append(learning_rate_change_steps, i)
    #             minimum_val_step_num = i
    #             checkpoint_manager.restore_or_initialize()

    # checkpoint_manager.restore_or_initialize()
    # # TODO: Butcher all traces of validation
    # current_validation_accuracy = val_accuracy(TransformerDecoder, val_text, val_labels)

    # fig, ax = plt.subplots(2, 1)

    # ax[0].plot(
    #     x_train_accuracy_iterations, y_train_batch_accuracy, label="Train Accuracy"
    # )
    # ax[0].plot(x_val_iterations, y_val_accuracy, label="Val Accuracy")
    # # plot vertical line on learning rate change
    # for learning_rate_change_step in learning_rate_change_steps:
    #     ax[0].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    # ax[0].set_xlabel("Iterations")
    # ax[0].set_ylabel("Accuracy")
    # ax[0].legend()

    # ax[1].semilogy(
    #     x_train_loss_iterations, y_train_batch_loss, label="Train Batch Loss"
    # )
    # ax[1].semilogy(x_val_iterations, y_val_loss, label="Val Loss")
    # for learning_rate_change_step in learning_rate_change_steps:
    #     ax[1].axvline(x=learning_rate_change_step, color="black", linestyle="dashed")
    # ax[1].set_xlabel("Iterations")
    # ax[1].set_ylabel("Loss")
    # ax[1].legend()

    # print("\n\n\n\n")
    # print(f"Final Training Loss => {current_train_batch_loss:0.4f}")
    # print(f"Stop Iteration => {i}")

    # fig.suptitle(
    #     "Predict Who Bites Who: Final Val Accuracy = "
    #     + f"{current_validation_accuracy:0.4f}"
    # )

    # # if the file already exists add a number to the end of the file name
    # # to avoid overwriting
    # file_index = 0
    # while Path(
    #     f"artifacts/who_bites_who/predict_who_bites_who_img_{file_index}.png"
    # ).exists():
    #     file_index += 1
    # fig.savefig(f"artifacts/who_bites_who/predict_who_bites_who_img_{file_index}.png")

    # # Save the config file as a yaml under the same name as the image
    # config_path = Path(
    #     f"artifacts/who_bites_who/predict_who_bites_who_img_{file_index}.yaml"
    # )
    # config_path.write_text(yaml.dump(config))

    # # save the model
    # checkpoint_manager.directory = "artifacts/who_bites_who/model"
    # checkpoint_manager.save()
    # config_path = Path(f"artifacts/who_bites_who/model.yaml")
    # config_path.write_text(yaml.dump(config))


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
    file.close()
    file.close()
    file.close()
    file.close()
    file.close()
