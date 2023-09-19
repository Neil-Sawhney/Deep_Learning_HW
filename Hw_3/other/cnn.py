import tensorflow as tf
import gzip
import numpy as np
from pathlib import Path
from tqdm import trange
import yaml
import argparse


class Conv2D(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        self.kernel = tf.Variable(
            tf.random.normal(
                [kernel_size[0], kernel_size[1], input_channels, output_channels]
            )
        )
        self.bias = tf.Variable(tf.zeros([output_channels]))
    
    def __call__(self, x):
        return tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding='SAME') + self.bias


class Classifier(tf.Module):
    def __init__(self, input_depth, layer_depths, layer_kernel_sizes, num_classes):
        self.layers = []
        
        current_depth = input_depth
        for depth, kernel_size in zip(layer_depths, layer_kernel_sizes):
            self.layers.append(Conv2D(current_depth, depth, kernel_size))
            current_depth = depth

        self.fc = tf.Variable(tf.random.normal([current_depth * 28 * 28, num_classes]))
        self.fc_bias = tf.Variable(tf.zeros([num_classes]))

    def __call__(self, x):
        for layer in self.layers:
            x = tf.nn.relu(layer(x))  
            x = tf.nn.dropout(x, rate=0.5)  
        x = tf.reshape(x, [-1, self.fc.shape[0]])
        return tf.matmul(x, self.fc) + self.fc_bias
    
def evaluate(model, x, y):
    logits = model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), dtype=tf.float32))
    return loss, accuracy

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28, 1)
    return data / np.float32(255)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)

x_train = load_mnist_images('train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('train-labels-idx1-ubyte.gz').astype(np.int32)
#x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
#y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

val_split = 0.1
val_count = int(val_split * len(x_train))
x_val, y_val = x_train[:val_count], y_train[:val_count]
x_train, y_train = x_train[val_count:], y_train[val_count:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST dataset")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    num_samples = x_train.shape[0]
    input_depth = x_train.shape[-1]
    num_classes = 10

    layer_depths = config["network"]["layer_depths"]
    layer_kernel_sizes = config["network"]["layer_kernel_sizes"]
    classifier = Classifier(input_depth, layer_depths, layer_kernel_sizes, num_classes)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    l2_reg_strength = .00001
    refresh_rate = config["display"]["refresh_rate"]

    rng = tf.random.get_global_generator()

    bar = trange(num_iters)
    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x_train, batch_indices)
            y_batch = tf.gather(y_train, batch_indices)

            logits = classifier(x_batch)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))

            l2_loss = 0
            for layer in classifier.layers:
                if isinstance(layer, Conv2D):
                    l2_loss += tf.nn.l2_loss(layer.kernel)
            loss += l2_reg_strength * l2_loss

            grads = tape.gradient(loss, classifier.trainable_variables)
            
            
            new_grads = []
            for grad, var in zip(grads, classifier.trainable_variables):
                if "kernel" in var.name:
                    new_grad = grad + 2.0 * l2_reg_strength * var
                    new_grads.append(new_grad)
                else:
                    new_grads.append(grad)

            grad_update(step_size, classifier.trainable_variables, new_grads)


            step_size *= decay_rate
            loss_float = float(loss.numpy().mean())
            if i % refresh_rate == (refresh_rate - 1):
                val_loss, val_accuracy = evaluate(classifier, x_val, y_val)
                bar.set_description(f"Step {i}; Train Loss: {loss_float:.4f}; Val Loss: {val_loss:.4f}; Val Accuracy: {val_accuracy:.4f}; Step Size: {step_size:.4f}")
                bar.refresh()