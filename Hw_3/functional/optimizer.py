import tensorflow as tf


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def adam(grads, variables, learning_rate):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = tf.Variable(0.0)
    for grad, var in zip(grads, variables):
        m = tf.Variable(tf.zeros(var.shape))
        v = tf.Variable(tf.zeros(var.shape))
        m.assign(beta1 * m + (1 - beta1) * grad)
        v.assign(beta2 * v + (1 - beta2) * grad * grad)
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        var.assign(var - learning_rate * m_hat / (tf.sqrt(v_hat) + epsilon))
    t.assign_add(1.0)
