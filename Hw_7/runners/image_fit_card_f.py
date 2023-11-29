import tempfile
from pathlib import Path

import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import yaml

from helpers.adam import Adam
from modules.siren_mlp import SirenMLP


def train(config_path: Path, use_last_checkpoint: bool):
    if config_path is None:
        config_path = Path("configs/image_fit_card_f.yaml")

    # HYPERPARAMETERS
    config = yaml.safe_load(config_path.read_text())
    refresh_rate = config["display"]["refresh_rate"]
    learning_patience = config["learning"]["learning_patience"]
    learning_rates = config["learning"]["learning_rates"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    num_hidden_layers = config["siren"]["num_hidden_layers"]
    hidden_layer_width = config["siren"]["hidden_layer_width"]
    siren_resolution = config["siren"]["resolution"]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    siren = SirenMLP(
        2,
        3,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_width=hidden_layer_width,
        hidden_activation=tf.math.sin,
        output_activation=tf.math.sin,
    )

    # Load the image
    input_image = cv2.imread("data/TestCardF.jpg")

    # Resize the image
    resized_img = cv2.resize(input_image, (siren_resolution, siren_resolution))

    # normalize the image
    img = resized_img / 255

    target = einops.rearrange(img, "h w c -> (h w) c")

    resolution = img.shape[0]
    # Generate a linear space from -1 to 1 with the same size as the resolution
    tmp = np.linspace(-1, 1, resolution)

    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(tmp, tmp)

    # Reshape and concatenate x and y, and cast them to float32
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)
    img = tf.cast(tf.concat((x_reshaped, y_reshaped), 1), tf.float32)

    used_patience = 0
    minimum_train_loss = np.inf
    minimum_loss_step_num = 0

    # Used For Plotting
    y_train_batch_loss = np.array([])
    x_train_loss_iterations = np.array([])

    learning_rate_change_steps = np.array([])

    # Index of the current learning rate, used to change the learning rate
    # when the training loss stops improving
    learning_rate_index = 0
    adam = Adam(
        learning_rates[learning_rate_index],
        weight_decay=weight_decay,
    )

    # find the temp_dir with the prefix if it exists
    # otherwise create a new one
    temp_dir = None
    for temp_dir in Path(tempfile.gettempdir()).iterdir():
        if temp_dir.is_dir() and temp_dir.name.startswith("siren_"):
            break

    if not temp_dir.name.startswith("siren_"):
        temp_dir = tempfile.mkdtemp(prefix="siren_")

    checkpoint = tf.train.Checkpoint(siren)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        temp_dir,
        max_to_keep=1,
    )
    if use_last_checkpoint:
        print("\n\nRestoring from last checkpoint")
        checkpoint_manager.restore_or_initialize()

    overall_log = tqdm.tqdm(total=0, position=1, bar_format="{desc}")
    train_log = tqdm.tqdm(total=0, position=2, bar_format="{desc}")
    bar = tqdm.trange(num_iters, position=3)

    num_of_parameters = tf.math.add_n(
        [tf.math.reduce_prod(var.shape) for var in siren.trainable_variables]
    )
    print(f"\nNumber of Parameters => {num_of_parameters}")

    for i in bar:
        with tf.GradientTape() as tape:
            logits = siren(img)

            current_train_loss = tf.reduce_mean((logits - target) ** 2)

        # Print initial train batch loss
        if i == 0:
            print("\n\n\n\n")
            print(f"Initial Training Loss => {current_train_loss:0.4f}")

        grads = tape.gradient(current_train_loss, siren.trainable_variables)

        adam.apply_gradients(zip(grads, siren.trainable_variables))

        current_train_loss = current_train_loss.numpy()

        if current_train_loss < minimum_train_loss:
            minimum_train_loss = current_train_loss
            minimum_loss_step_num = i
            checkpoint_manager.save()
        used_patience = i - minimum_loss_step_num

        y_train_batch_loss = np.append(y_train_batch_loss, current_train_loss)
        x_train_loss_iterations = np.append(x_train_loss_iterations, i)

        if i % refresh_rate == (refresh_rate - 1):
            learning_rates_left = len(learning_rates) - learning_rate_index
            patience_left = learning_patience - used_patience
            overall_description = (
                f"Minimum Train Loss => {minimum_train_loss:0.4f}    "
                + f"Learning Rates Left => {learning_rates_left}    "
                + f"Patience Left => {patience_left}    "
            )
            overall_log.set_description_str(overall_description)
            overall_log.refresh()

            train_description = f"Train Batch Loss => {current_train_loss:0.4f}    "
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
                checkpoint_manager.restore_or_initialize()

    checkpoint_manager.restore_or_initialize()

    # delete the temporary directory
    tf.io.gfile.rmtree(temp_dir)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "artifacts/siren/model", max_to_keep=1
    )
    checkpoint_manager.save()

    fig = plt.figure(figsize=(10, 10))

    # Create a grid of 2 rows and 2 columns
    grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)

    # Use the grid to specify the location of each subplot
    main_ax = fig.add_subplot(grid[0, :])
    y_image_ax = fig.add_subplot(grid[1, 0])
    x_image_ax = fig.add_subplot(grid[1, 1])

    main_ax.semilogy(x_train_loss_iterations, y_train_batch_loss, label="Training Loss")
    for learning_rate_change_step in learning_rate_change_steps:
        main_ax.axvline(
            x=learning_rate_change_step,
            color="black",
            linestyle="dashed",
            label="Learning Rate Change",
        )
    main_ax.axvline(
        x=minimum_loss_step_num, color="red", linestyle="dashed", label="Minimum Loss"
    )
    main_ax.set_xlabel("Iterations")
    main_ax.set_ylabel("Loss")
    main_ax.legend()

    output_image = einops.rearrange(
        logits.numpy(), "(h w) c -> h w c", h=siren_resolution, w=siren_resolution
    )

    # Downscale the input image then rescale it back up to make it a fair comparison
    downscaled_input_image = cv2.resize(
        input_image, (siren_resolution, siren_resolution)
    )
    rescaled_input_image = cv2.resize(
        input_image, (input_image.shape[1], input_image.shape[0])
    )

    y_image_ax.imshow(
        cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), interpolation="none"
    )
    y_image_ax.set_title("Ground Truth")
    y_image_ax.axis("off")

    # reshape output to input image shape
    output_image = cv2.resize(
        output_image, (input_image.shape[1], input_image.shape[0])
    )

    x_image_ax.imshow(
        cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), interpolation="none"
    )
    x_image_ax.set_title("Prediction")
    x_image_ax.axis("off")

    fig.suptitle("Siren - Card F: Image Fitting")

    plt.show()

    print("\n\n\n\n")

    print(f"Stop Iteration => {i}")

    # if the file already exists add a number to the end of the file name
    # to avoid overwriting
    file_index = 0
    while Path(f"artifacts/siren/siren_img_{file_index}.png").exists():
        file_index += 1
    fig.savefig(f"artifacts/siren/siren_img_{file_index}.png")

    # Save the config file as a yaml under the same name as the image
    config_path = Path(f"artifacts/siren/siren_img_{file_index}.yaml")
    config_path.write_text(yaml.dump(config))

    # save the model
    checkpoint_manager.save()
    config_path = Path(f"artifacts/siren/model/model.yaml")
    config_path.write_text(yaml.dump(config))


# def test(model_path: Path):
#     if model_path is None:
#         model_path = Path("artifacts/who_bites_who/model")

#     if not model_path.exists():
#         print("Model does not exist, run the train script first")
#         return

#     config_path = Path("artifacts/who_bites_who/model/model.yaml")

#     config = yaml.safe_load(config_path.read_text())
#     context_length = config["data"]["context_length"]
#     num_heads = config["transformer"]["num_heads"]
#     model_dim = config["transformer"]["model_dim"]
#     ffn_dim = config["transformer"]["ffn_dim"]
#     num_blocks = config["transformer"]["num_blocks"]

#     rng = tf.random.get_global_generator()
#     rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
#     tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

#     vocab_file = Path("artifacts/who_bites_who/model/vocab.txt")

#     transformer_decoder = TransformerDecoder(
#         context_length,
#         num_heads,
#         model_dim,
#         ffn_dim,
#         num_blocks,
#         vocab_file=vocab_file,
#     )

#     checkpoint = tf.train.Checkpoint(transformer_decoder)
#     checkpoint.restore(tf.train.latest_checkpoint(model_path))

#     print("\n\n\n\nEnter 'exit' to exit")
#     while 1:
#         input_text = input("\n\nEnter a sentence: ")
#         if input_text == "exit":
#             break

#         tokenized_text = transformer_decoder.predict(input_text)
#         print(f"Bite Bot: " + tokenized_text)
