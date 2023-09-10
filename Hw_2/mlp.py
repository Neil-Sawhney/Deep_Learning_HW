import tensorflow as tf

from linear import Linear


class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers,
                 hidden_layer_width, hidden_activation=tf.identity,
                 output_activation=tf.identity):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_linear = Linear(self.hidden_layer_width,
                                    self.hidden_layer_width)
        self.first_linear = Linear(num_inputs, hidden_layer_width)
        self.final_linear = Linear(self.hidden_layer_width, self.num_outputs)

    def __call__(self, x):
        x = self.hidden_activation(self.first_linear(x))
        for i in range(self.num_hidden_layers):
            x = self.hidden_activation(
                self.hidden_linear(x)
            )
        return self.output_activation(self.final_linear(x))


if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange
    from linear import grad_update
    from math import pi

    parser = argparse.ArgumentParser(
        prog="Multi Layer Perceptron",
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

    num_samples = config["data"]["num_samples"]
    num_inputs = 2
    num_outputs = 1

    noise_stddev = config["data"]["noise_stddev"]
    e1 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    e2 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    r = tf.linspace(0., 4*pi, num_samples)[:, tf.newaxis]

    x1 = r*tf.math.cos(r) + e1
    y1 = r*tf.math.sin(r) + e1
    x2 = -r*tf.math.cos(r) + e2
    y2 = -r*tf.math.sin(r) + e2

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    def relu(x):
        return tf.maximum(x, 0)

    def sigmoid(x):
        return tf.math.sigmoid(x)

    # TODO: do we want an output activation?
    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width,
              relu, sigmoid)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size*2], maxval=num_samples*2, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            output1 = tf.zeros((num_samples, 1))
            output2 = tf.ones((num_samples, 1))

            input1 = tf.concat((x1, y1), axis=1)
            input2 = tf.concat((x2, y2), axis=1)

            output_combined = tf.concat((output1, output2), axis=0)
            input_combined = tf.concat((input1, input2), axis=0)

            output_batch = tf.gather(output_combined, batch_indices)
            input_batch = tf.gather(input_combined, batch_indices)

            y_hat = mlp(input_batch)

            loss = tf.math.reduce_mean(
                -output_batch*tf.math.log(y_hat + 1e-7) -
                (1 - output_batch)*tf.math.log(1 - y_hat + 1e-7)
            )

        grads = tape.gradient(loss, mlp.trainable_variables)
        grad_update(step_size, mlp.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, \
                    step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, ax = plt.subplots()

    ax.plot(x1.numpy().squeeze(), y1.numpy().squeeze(), "x", label="Inputs")
    ax.plot(x2.numpy().squeeze(), y2.numpy().squeeze(), "x", label="Inputs")

    x1_quiet = r*tf.math.cos(r)
    y1_quiet = r*tf.math.sin(r)
    ax.plot(
        x1_quiet.numpy().squeeze(),
        y1_quiet.numpy().squeeze(),
        "-",
        label="Clean"
    )

    x2_quiet = -r*tf.math.cos(r)
    y2_quiet = -r*tf.math.sin(r)
    ax.plot(
        x2_quiet.numpy().squeeze(),
        y2_quiet.numpy().squeeze(),
        "-",
        label="Clean"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Nonlinear fit using SGD")

    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    import sklearn.inspection.DecisionBoundaryDisplay as dbd

    fig.savefig("artifacts/mlp.pdf")
