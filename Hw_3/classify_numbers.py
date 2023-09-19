import argparse
from pathlib import Path

import tensorflow as tf
import yaml
from tqdm import trange

from functional.idx_loader import load_idx_data
from functional.optimizer import grad_update
from models.classifier import Classifier

parser = argparse.ArgumentParser(
    prog="CNN",
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

train_images = load_idx_data("data/train-images-idx3-ubyte")[:-10000]
train_labels = load_idx_data("data/train-labels-idx1-ubyte")[:-10000]
val_images = load_idx_data("data/train-images-idx3-ubyte")[-10000:]
val_labels = load_idx_data("data/train-labels-idx1-ubyte")[-10000:]
test_images = load_idx_data("data/t10k-images-idx3-ubyte")
test_labels = load_idx_data("data/t10k-labels-idx1-ubyte")

input_depth = 1
layer_depths = config["cnn"]["layer_depths"]
kernel_sizes = config["cnn"]["kernel_sizes"]
num_iters = config["learning"]["num_iters"]
step_size = config["learning"]["step_size"]
decay_rate = config["learning"]["decay_rate"]
l2_scale = config["learning"]["l2_scale"]
dropout_prob = config["learning"]["dropout_prob"]
refresh_rate = config["display"]["refresh_rate"]

classifier = Classifier(input_depth,
                        layer_depths,
                        kernel_sizes,
                        10,
                        train_images.shape[1],
                        2,
                        2,
                        dropout_prob)

bar = trange(num_iters)

for i in bar:
    with tf.GradientTape() as tape:
        kernel_weights = []
        for layer in classifier.conv_layers:
            kernel_weights.append(layer.kernel)
        l2_loss = tf.nn.l2_loss(kernel_weights)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels,
            logits=classifier(train_images)
        )) + l2_scale * l2_loss

    grads = tape.gradient(loss, classifier.trainable_variables)
    grad_update(step_size, classifier.trainable_variables, grads)

    step_size *= decay_rate

    if i % refresh_rate == (refresh_rate - 1):
        bar.set_description(
            f"Step {i}; Loss => {loss.numpy():0.4f}, \
                step_size => {step_size:0.4f}"
        )
        bar.refresh()
