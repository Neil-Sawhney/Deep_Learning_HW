import tensorflow as tf


class Adam:
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=5e-3,):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def apply_gradients(
        self,
        grads_and_vars
    ):
        for grad, var in grads_and_vars:
            m = tf.Variable(tf.zeros(shape=var.shape))
            v = tf.Variable(tf.zeros(shape=var.shape))
            m.assign(self.beta_1 * m + (1 - self.beta_1) * tf.convert_to_tensor(grad))
            v.assign(self.beta_2 * v + (1 - self.beta_2) * tf.convert_to_tensor(grad) ** 2)
            m_hat = m / (1 - self.beta_1)
            v_hat = v / (1 - self.beta_2)
            var.assign(var - self.learning_rate * m_hat /
                       (tf.sqrt(v_hat) + self.epsilon))
            if (var.name.endswith('kernel') or var.name.endswith('w')):
                var.assign(var - self.weight_decay * var * self.learning_rate)
