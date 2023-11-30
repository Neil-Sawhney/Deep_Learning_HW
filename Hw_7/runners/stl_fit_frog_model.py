import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import yaml
from stl import mesh

from helpers.adam import Adam
from modules.siren_mlp import SirenMLP


def train(config_path: Path, use_last_checkpoint: bool):
    if config_path is None:
        config_path = Path("configs/stl_fit_frog_model.yaml")

    # HYPERPARAMETERS
    config = yaml.safe_load(config_path.read_text())
    refresh_rate = config["display"]["refresh_rate"]
    learning_patience = config["learning"]["learning_patience"]
    learning_rates = config["learning"]["learning_rates"]
    num_iters = config["learning"]["num_iters"]
    weight_decay = config["learning"]["weight_decay"]
    num_hidden_layers = config["siren"]["num_hidden_layers"]
    hidden_layer_width = config["siren"]["hidden_layer_width"]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    siren = SirenMLP(
        3,
        3,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_width=hidden_layer_width,
        hidden_activation=tf.math.sin,
        output_activation=tf.math.sin,
    )

    # Load the mesh
    input_mesh = mesh.Mesh.from_file("data/frog.stl")

    # Convert the mesh to an array of vectors
    input_mesh_vect = np.array(input_mesh.vectors)

    # Get the maximum value of the mesh
    max_value = np.amax(input_mesh_vect)

    # normalize the mesh
    input_mesh_vect = input_mesh_vect / max_value

    # Generate a tensor of value 1 with the same shape as the input mesh
    mesh_template = np.ones(input_mesh_vect.shape)

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
        if temp_dir.is_dir() and temp_dir.name.startswith("siren_fit_model_"):
            break

    if not temp_dir.name.startswith("siren_fit_model_"):
        temp_dir = tempfile.mkdtemp(prefix="siren_fit_model_")

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
            logits = siren(mesh_template)

            current_train_loss = tf.reduce_mean((logits - input_mesh_vect) ** 2)

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

    # save the predicted mesh
    predicted_mesh = mesh.Mesh(
        np.zeros(input_mesh.vectors.shape[0], dtype=mesh.Mesh.dtype)
    )
    predicted_mesh.vectors = siren(mesh_template).numpy() * max_value
    predicted_mesh.save("artifacts/siren_fit_model/predicted_frog.stl")

    # delete the temporary directory
    tf.io.gfile.rmtree(temp_dir)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "artifacts/siren_fit_model/model", max_to_keep=1
    )
    checkpoint_manager.save()

    fig, main_ax = plt.subplots()

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

    fig.suptitle("Siren - Frog: Model Fitting")

    plt.show()

    print("\n\n\n\n")

    print(f"Stop Iteration => {i}")

    # if the file already exists add a number to the end of the file name
    # to avoid overwriting
    file_index = 0
    while Path(f"artifacts/siren_fit_model/siren_img_{file_index}.png").exists():
        file_index += 1
    fig.savefig(f"artifacts/siren_fit_model/siren_img_{file_index}.png")

    # Save the config file as a yaml under the same name as the image
    config_path = Path(f"artifacts/siren_fit_model/siren_img_{file_index}.yaml")
    config_path.write_text(yaml.dump(config))

    # save the model
    checkpoint_manager.save()
    config_path = Path(f"artifacts/siren_fit_model/model/model.yaml")
    config_path.write_text(yaml.dump(config))
