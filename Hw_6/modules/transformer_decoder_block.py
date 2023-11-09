import tensorflow as tf

from helpers.group_norm import GroupNorm
from modules.mlp import MLP
from modules.multi_head_attention import MultiHeadAttention


class TransformerDecoderBlock(tf.Module):
    """Transformer Decoder Block.

    This is an implementation of a single transformer decoder block as described
    in the paper "Attention is all you Need" (Vaswani et al., 2017).

    This layer first applies masked multi-headed attention to the inputs, then applies a feed-forward network to the result.

    Args:
        num_heads: Number of attention heads.
        model_dim: Size of each attention head for value, query, and queue.
        ffn_dim: Size of the hidden layer in the feed-forward network.
        dropout: Dropout probability.

    Call arguments:
        inputs: Input `Tensor` of shape `(B, seq_len, model_dim)`.
        mask: Optional mask tensor of shape `(B, seq_len, seq_len)`.
    """

    def __init__(self, num_heads, model_dim, ffn_dim, dropout_prob=0.1):
        self.dropout_prob = dropout_prob

        self.mha = MultiHeadAttention(num_heads, model_dim)
        self.ff = MLP(
            model_dim,
            model_dim,
            hidden_layer_width=ffn_dim,
            hidden_activation=tf.nn.relu,
        )
        self.groupnorm1 = GroupNorm(num_groups=1, input_depth=model_dim)
        self.groupnorm2 = GroupNorm(num_groups=1, input_depth=model_dim)
        self.groupnorm3 = GroupNorm(num_groups=1, input_depth=model_dim)

    def __call__(self, inputs, mask=False, training=False):
        attn = self.mha(inputs, inputs, inputs, mask)
        if training:
            attn = tf.nn.dropout(attn, rate=self.dropout_prob)
        out1 = self.groupnorm1(attn + inputs)

        ffn_output = self.ff(out1)
        if training:
            ffn_output = tf.nn.dropout(ffn_output, rate=self.dropout_prob)
        out2 = self.groupnorm3(ffn_output + out1)

        return out2
