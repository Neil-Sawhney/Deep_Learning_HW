import tensorflow as tf

from linear import Linear


class MLP(tf.module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers,
                 hidden_layer_width, hidden_activation=tf.identity,
                 output_activation=tf.identity):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.Linear = Linear(num_inputs, num_outputs)

    def __call__(self, x):
        for i in range(self.num_hidden_layers):
            x = self.hidden_activation(
                self.Linear(self.num_inputs,
                            self.hidden_layer_width)(x)
            )
        return self.output_activation(x)


if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange
    from linear import grad_update

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
    num_inputs = 1
    num_outputs = 1

    x = rng.uniform(shape=(num_samples, num_inputs))
    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))
    y = rng.normal(
        shape=(num_samples, num_outputs),
        mean=x @ w + b,
        stddev=config["data"]["noise_stddev"],
    )

    linear = Linear(num_inputs, num_outputs)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            y_hat = linear(x_batch)
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(loss, linear.trainable_variables)
        grad_update(step_size, linear.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, \
                    step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, ax = plt.subplots()

    ax.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")

    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax.plot(a.numpy().squeeze(), linear(a).numpy().squeeze(), "-")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear fit using SGD")
    
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    fig.savefig("./artifacts/linear.pdf")
