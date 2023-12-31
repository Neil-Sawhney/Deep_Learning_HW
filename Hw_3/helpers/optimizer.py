import tensorflow as tf


class Adam:
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=5e-3):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def apply_gradients(
        self,
        grads, vars
    ):
        for grad, var in zip(grads, vars):
            m = tf.Variable(tf.zeros(shape=var.shape))
            v = tf.Variable(tf.zeros(shape=var.shape))
            m.assign(self.beta_1 * m + (1 - self.beta_1) * grad)
            v.assign(self.beta_2 * v + (1 - self.beta_2) * grad ** 2)
            m_hat = m / (1 - self.beta_1)
            v_hat = v / (1 - self.beta_2)
            var.assign(var - self.learning_rate * m_hat /
                       (tf.sqrt(v_hat) + self.epsilon))
        weights = [weight for weight in vars if
                   weight.name.endswith('kernel')]
        for weight in weights:
            weight.assign(weight - self.weight_decay * weight)
