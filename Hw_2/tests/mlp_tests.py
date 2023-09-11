import pytest


def test_non_additivity():
    import tensorflow as tf

    from mlp import MLP

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

    from mlp import MLP

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

    from mlp import MLP

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

    from mlp import MLP

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

    from mlp import MLP

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


# ensure that all the gradients are greater than 0
def test_trainable():
    import tensorflow as tf

    from mlp import MLP

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


if __name__ == "__main__":
    pytest.main([__file__])
