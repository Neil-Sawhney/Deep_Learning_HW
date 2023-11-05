import tensorflow as tf

from modules.linear import Linear

mask = tf.fill([bs, num_heads, seq_length, d_model // num_heads], float("-inf"))
mask = tf.linalg.band_part(mask, 0, -1)
mask = tf.linalg.set_diag(mask, tf.zeros([bs, num_heads, seq_length]))


class MultiHeadAttention(tf.Module):
    def __init__(self, num_heads, d_model):
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.n_heads = num_heads

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

    def __call__(self, input: tf.Tensor, mask: tf.Tensor = None):
        Q = self.wq(input)
        print(Q.shape)
        K = self.wk(input)
        print(K.shape)
        V = self.wv(input)
        print(V.shape)

        batch_size = Q.shape[0]

        # split into n-heads
        Q = tf.transpose(
            tf.reshape(Q, (batch_size, -1, self.n_heads, self.d_k)), (0, 2, 1, 3)
        )  # [bs,len,dm] -> [bs,len,nh,dk]
        print(Q.shape)
        K = tf.transpose(
            tf.reshape(K, (batch_size, -1, self.n_heads, self.d_k)), (0, 2, 1, 3)
        )  # [bs,len,dm] -> [bs,len,nh,dk]
        print(K.shape)
        V = tf.transpose(
            tf.reshape(V, (batch_size, -1, self.n_heads, self.d_k)), (0, 2, 1, 3)
        )  # [bs,len,dm] -> [bs,len,nh,dk]
        print(V.shape)
        # SDP = QK^T
        scaled_dot_prod = einops.einsum(Q, K, "b s i k, b s j k -> b s i k") / np.sqrt(
            self.d_k
        )
        print(scaled_dot_prod.shape)

        if mask is not None:
            print(mask)
            # breakpoint()
            scaled_dot_prod += mask

        attention = tf.nn.softmax(scaled_dot_prod, -1)
        print(attention)

        A = einops.einsum(attention, V, "b s i k, b s j k  -> b s i k")
        A = tf.reshape(
            (tf.transpose(A, (0, 2, 1, 3))), (batch_size, -1, self.n_heads * self.d_k)
        )

        output = self.wo(A)
        # breakpoint()
        return output, attention
