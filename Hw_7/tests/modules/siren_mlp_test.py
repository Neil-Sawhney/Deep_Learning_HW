import pytest


def test_initialization_loss():
    import cv2
    import einops
    import numpy as np
    import tensorflow as tf

    from modules.siren_mlp import SirenMLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_outputs = 3
    num_hidden_layers = 10
    hidden_layer_width = 10
    siren_resolution = 256

    siren = SirenMLP(
        num_inputs,
        num_outputs,
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

    logits = siren(img)

    initial_training_loss = tf.reduce_mean(logits)

    tf.debugging.assert_near(initial_training_loss, 0, atol=0.001)


def test_non_additivity():
    import tensorflow as tf

    from modules.siren_mlp import SirenMLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10
    relu = tf.nn.relu
    sigmoid = tf.nn.sigmoid

    mlp = SirenMLP(
        num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, relu, sigmoid
    )

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    case1 = mlp(a + b)
    case2 = mlp(a) + mlp(b)

    tol = 2.22e-15 + 2.22e-15 * tf.abs(case2)

    tf.debugging.Assert(
        tf.reduce_any(tf.greater(tf.abs(case1 - case2), tol)),
        [case1, case2],
        summarize=2,
    )


def test_non_homogeneity():
    import tensorflow as tf

    from modules.siren_mlp import SirenMLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100
    num_hidden_layers = 10
    hidden_layer_width = 10
    relu = tf.nn.relu
    sigmoid = tf.nn.sigmoid

    mlp = SirenMLP(
        num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, relu, sigmoid
    )

    a = rng.normal(shape=[1, 1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1, 1])

    case1 = mlp(a * b)
    case2 = mlp(a) * b

    tol = 2.22e-15 + 2.22e-15 * tf.abs(case2)

    tf.debugging.Assert(
        tf.reduce_any(tf.greater(tf.abs(case1 - case2), tol)),
        [case1, case2],
        summarize=2,
    )


def test_dimensionality():
    import tensorflow as tf

    from modules.siren_mlp import SirenMLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10

    mlp = SirenMLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])
    z = mlp(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


def test_trainable():
    import tensorflow as tf

    from modules.siren_mlp import SirenMLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10

    mlp = SirenMLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = mlp(a)
        loss = tf.reduce_mean(z**2)

    grads = tape.gradient(loss, mlp.trainable_variables)

    for grad, var in zip(grads, mlp.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(mlp.trainable_variables)


if __name__ == "__main__":
    pytest.main([__file__])
