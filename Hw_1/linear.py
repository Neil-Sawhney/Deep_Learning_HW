#!/bin/env python

import math

import tensorflow as tf

from basisExpansion import BasisExpansion


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        # initialize weights to random values from a normal distribution
        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    # create the logits by multiplying the inputs by the weights + the
    # optional bias
    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
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
    noise_stddev = config["data"]["noise_stddev"]
    num_bases = config["model"]["num_bases"]
    num_inputs = 1
    num_outputs = 1

    x = rng.uniform(shape=(num_samples, num_inputs))
    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))
    epsilon_noisy = rng.normal(
        shape=(num_samples, num_inputs),
        stddev=noise_stddev)
    y = tf.math.sin(2*tf.constant(math.pi)*x) + epsilon_noisy

    basisExpansion = BasisExpansion(num_bases, num_inputs, num_outputs)
    linear = Linear(num_bases, num_outputs)

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

            y_hat = linear(basisExpansion(x_batch))
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        trainables = (linear.trainable_variables
                      + basisExpansion.trainable_variables)

        grads = tape.gradient(loss, trainables)
        grad_update(step_size, trainables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, "
                f"step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    ax.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x", label="Inputs")

    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    y_quiet = tf.math.sin(2*tf.constant(math.pi)*a)
    ax.plot(a.numpy().squeeze(), y_quiet.numpy().squeeze(), "-", label="Clean")

    ax.plot(
        a.numpy().squeeze(),
        linear(basisExpansion(a)).numpy().squeeze(),
        "-",
        label="Fit"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Nonlinear fit using SGD")

    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    a2 = tf.linspace(-3., 3., 1000)[:, tf.newaxis]
    ax2.plot(a2.numpy().squeeze(), basisExpansion(a2).numpy().squeeze(), "-")

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Basis functions")

    fig.savefig("artifacts/plot.pdf")
