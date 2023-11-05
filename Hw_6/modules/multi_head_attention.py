import einops
import tensorflow as tf

from modules.linear import Linear


class MultiHeadAttention(tf.Module):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017).

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_heads`, where the
    corresponding shapes are `(batch_size, seq_len, model_dim)`

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as model_dim can take an
    linear projection and return.

    Args:
        num_heads: Number of attention heads.
        model_dim: Size of each attention head for value, query, and queue.
        dropout: Dropout probability.

    Call arguments:
        query: Query `Tensor` of shape `(B, seq_len, model_dim)`.
        value: Value `Tensor` of shape `(B, seq_len, model_dim)`.
        key: Optional key `Tensor` of shape `(B, seq_len, model_dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.

    Returns:
        output: The result of the computation, of shape `(B, seq_len, model_dim)`,
        attention_weights: multi-head attention coefficients
    """

    def __init__(self, num_heads, model_dim, dropout=0.1):
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout

        assert model_dim % num_heads == 0

        self.depth = model_dim // num_heads

        self.wq = Linear(model_dim, model_dim)
        self.wk = Linear(model_dim, model_dim)
        self.wv = Linear(model_dim, model_dim)
        self.wo = Linear(model_dim, model_dim)

    def _split_heads(self, inputs):
        """Split the last dimension into (num_heads, depth). Transpose and organize the result

        Args:
            inputs: input tensor of shape `(batch_size, seq_len, model_dim)`
            batch_size: batch size

        Returns:
            A tensor with shape `(batch_size, num_heads, seq_len, depth)`
        """
        output = einops.rearrange(
            inputs,
            "batch seq (heads depth) -> batch heads seq depth",
            heads=self.num_heads,
        )

        return output

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled dot product attention

        Args:
            q: query tensor of shape `(batch_size, num_heads, seq_len, depth)`
            k: key tensor of shape `(batch_size, num_heads, seq_len, depth)`
            v: value tensor of shape `(batch_size, num_heads, seq_len, depth)`
            mask: optional mask tensor of shape `(batch_size, seq_len, seq_len)`

        Returns:
            A tuple of two tensors:
                output: output tensor of shape `(batch_size, num_heads, seq_len, depth)`
                attention_weights: attention weights tensor of shape `(batch_size, num_heads, seq_len, seq_len)`
        """
        # matmul q and k while transposing k: (batch_size, num_heads, seq_len, seq_len)
        matmul_qk = tf.einsum(
            "batch heads q_seq depth, batch heads k_seq depth -> batch heads q_seq k_seq",
            q,
            k,
        )
        depth = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            # we want -inf where mask is 1 because of the softmax
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.einsum(
            "batch heads q_seq k_seq, batch heads v_seq depth -> batch heads q_seq depth",
            attention_weights,
            v,
        )

        return output, attention_weights

    def __call__(self, query, value, key=None, use_causal_mask=False):
        key = value if key is None else key

        if use_causal_mask:
            # create a lower triangular mask of shape (batch_size, seq_len, seq_len)
            mask = tf.linalg.band_part(
                tf.ones((query.shape[0], query.shape[1], query.shape[1])), -1, 0
            )

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        scaled_attention, attention_weights = self._scaled_dot_product_attention(
            query, key, value, mask
        )

        scaled_attention = einops.rearrange(
            scaled_attention, "batch heads seq depth -> batch seq (heads depth)"
        )

        output = self.wo(scaled_attention)

        return output, attention_weights
