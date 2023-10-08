import pytest


def test_non_additivity():
    import tensorflow as tf

    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10
    relu = tf.nn.relu
    sigmoid = tf.nn.sigmoid

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width,
              relu, sigmoid)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    case1 = mlp(a + b)
    case2 = mlp(a) + mlp(b)

    tol = 2.22e-15 + 2.22e-15*tf.abs(case2)

    tf.debugging.Assert(
        tf.reduce_any(
            tf.greater(
                tf.abs(
                    case1 - case2
                    ),
                tol
                )
            ),
        [case1, case2],
        summarize=2
    )


def test_non_homogeneity():
    import tensorflow as tf

    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100
    num_hidden_layers = 10
    hidden_layer_width = 10
    relu = tf.nn.relu
    sigmoid = tf.nn.sigmoid

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width,
              relu, sigmoid)

    a = rng.normal(shape=[1, 1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1, 1])

    case1 = mlp(a * b)
    case2 = mlp(a) * b

    tol = 2.22e-15 + 2.22e-15*tf.abs(case2)

    tf.debugging.Assert(
        tf.reduce_any(
            tf.greater(
                tf.abs(
                    case1 - case2
                    ),
                tol
                )
            ),
        [case1, case2],
        summarize=2
    )


def test_additivity():
    import tensorflow as tf

    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(mlp(a + b), mlp(a) + mlp(b), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100
    num_hidden_layers = 10
    hidden_layer_width = 10

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(mlp(a * b), mlp(a) * b, summarize=2)


def test_dimensionality():
    import tensorflow as tf

    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])
    z = mlp(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


def test_trainable():
    import tensorflow as tf

    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 10
    hidden_layer_width = 10

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = mlp(a)
        loss = tf.reduce_mean(z**2)

    grads = tape.gradient(loss, mlp.trainable_variables)

    for grad, var in zip(grads, mlp.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(mlp.trainable_variables)


def test_classification():
    from math import pi

    import tensorflow as tf

    from runners.classify_numbers import grad_update
    from modules.mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_outputs = 1

    import yaml
    config = yaml.safe_load(open("config.yaml"))
    num_iterations = config["learning"]["num_iters"]
    batch_size = config["learning"]["batch_size"]
    num_samples = config["data"]["num_samples"]
    step_size = config["learning"]["step_size"]
    noise_stddev = config["data"]["noise_stddev"]
    l2_scale = config["learning"]["l2_scale"]
    decay_rate = config["learning"]["decay_rate"]
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width,
              tf.nn.relu, tf.nn.sigmoid)

    e1 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    e2 = rng.normal(shape=(num_samples, 1), stddev=noise_stddev)
    r = tf.linspace(pi/4, 4*pi, num_samples)[:, tf.newaxis]

    r = tf.linspace(pi/4, 4*pi, num_samples)[:, tf.newaxis]
    x1 = (r + e1)*tf.math.cos(r)
    y1 = (r + e1)*tf.math.sin(r)
    x2 = -(r + e2)*tf.math.cos(r)
    y2 = -(r + e2)*tf.math.sin(r)

    for i in range(num_iterations):
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

    # check that mlp classifies all the points in spiral 1 and 2 corerctly
    x_spiral1 = (r)*tf.math.cos(r)
    y_spiral1 = (r)*tf.math.sin(r)
    test_input1 = tf.concat((x_spiral1, y_spiral1), axis=1)
    test_output1 = mlp(test_input1)

    tf.debugging.assert_less(test_output1, tf.fill(test_output1.shape, 0.5))

    x_spiral2 = -(r)*tf.math.cos(r)
    y_spiral2 = -(r)*tf.math.sin(r)
    test_input2 = tf.concat((x_spiral2, y_spiral2), axis=1)
    test_output2 = mlp(test_input2)

    tf.debugging.assert_greater(test_output2, tf.fill(test_output2.shape, 0.5))


if __name__ == "__main__":
    pytest.main([__file__])
