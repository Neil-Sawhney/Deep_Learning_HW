<<<<<<< HEAD
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
    from math import pi
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
    num_inputs = 2
    num_outputs = 1

    noise_stddev = config["data"]["noise_stddev"]
    e1 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    e2 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    r = tf.linspace(pi/4, 4*pi, num_samples)[:, tf.newaxis]

    x1 = (r + e1)*tf.math.cos(r)
    y1 = (r + e1)*tf.math.sin(r)
    x2 = -(r + e2)*tf.math.cos(r)
    y2 = -(r + e2)*tf.math.sin(r)

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    def relu(x):
        return tf.maximum(x, 0)

    def sigmoid(x):
        return tf.math.sigmoid(x)

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width,
              relu, sigmoid)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    l2_scale = config["learning"]["l2_scale"]

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

            weightTensor = tf.concat([tf.reshape(weight, [-1]) for weight in
                                      mlp.trainable_variables if weight.name ==
                                      "Linear/w:0"], axis=0)

            l2_norm = tf.norm(weightTensor, ord=2)
            loss = tf.math.reduce_mean(
                -output_batch*tf.math.log(y_hat + 1e-7) -
                (1 - output_batch)*tf.math.log(1 - y_hat + 1e-7)
            ) + l2_scale*l2_norm

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
    ax.set_title("Spiral Classification with Multi Layer Perceptron")

    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    import numpy as np
    from sklearn.inspection import DecisionBoundaryDisplay
    xx, yy = np.meshgrid(
        np.linspace(-15., 15., 100),
        np.linspace(-15., 15., 100)
    )

    input_grid = tf.stack(tf.convert_to_tensor((xx.ravel(), yy.ravel()),
                                               dtype=tf.float32), axis=1,)
    zz = mlp(input_grid)
    zz = zz.numpy().reshape(xx.shape)

    dbd = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=zz)
    dbd.plot(ax=ax, cmap="RdBu", alpha=0.9)

    fig.savefig("artifacts/mlp.pdf")
=======
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
    from math import pi
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
    num_inputs = 2
    num_outputs = 1

    noise_stddev = config["data"]["noise_stddev"]
    e1 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    e2 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    r = tf.linspace(pi/4, 4*pi, num_samples)[:, tf.newaxis]

    x1 = (r + e1)*tf.math.cos(r)
    y1 = (r + e1)*tf.math.sin(r)
    x2 = -(r + e2)*tf.math.cos(r)
    y2 = -(r + e2)*tf.math.sin(r)

    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    def relu(x):
        return tf.maximum(x, 0)

    def sigmoid(x):
        return tf.math.sigmoid(x)

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width,
              relu, sigmoid)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    l2_scale = config["learning"]["l2_scale"]

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

            weightTensor = tf.concat([tf.reshape(weight, [-1]) for weight in
                                      mlp.trainable_variables if weight.name ==
                                      "Linear/w:0"], axis=0)

            l2_norm = tf.norm(weightTensor, ord=2)
            loss = tf.math.reduce_mean(
                -output_batch*tf.math.log(y_hat + 1e-7) -
                (1 - output_batch)*tf.math.log(1 - y_hat + 1e-7)
            ) + l2_scale*l2_norm

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

    import numpy as np
    from sklearn.inspection import DecisionBoundaryDisplay
    xx, yy = np.meshgrid(
        np.linspace(-15., 15., 100),
        np.linspace(-15., 15., 100)
    )

    input_grid = tf.stack(tf.convert_to_tensor((xx.ravel(), yy.ravel()),
                                               dtype=tf.float32), axis=1,)
    zz = mlp(input_grid)
    zz = zz.numpy().reshape(xx.shape)

    dbd = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=zz)
    dbd.plot(ax=ax, cmap="RdBu", alpha=0.9)

    fig.savefig("artifacts/mlp.pdf")
>>>>>>> f236273fa2296cdaf19788c775b21b0c97dcd388
