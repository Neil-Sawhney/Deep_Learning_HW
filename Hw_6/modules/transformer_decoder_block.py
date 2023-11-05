from typing import Any

import tensorflow as tf

from helpers.group_norm import GroupNorm
from modules.mlp import MLP
from modules.multi_head_attention import MultiHeadAttention


class TransformerDecoderBlock(tf.Module):
    def __init__(self, num_heads, model_dim, ffn_dim, dropout_prob=0.1):
        self.dropout_prob = dropout_prob

        self.mha1 = MultiHeadAttention(num_heads, model_dim)
        self.mha2 = MultiHeadAttention(num_heads, model_dim)
        self.ff = MLP(ffn_dim, model_dim, hidden_activation=tf.nn.relu)
        self.groupnorm1 = GroupNorm(model_dim)
        self.groupnorm2 = GroupNorm(model_dim)
        self.groupnorm3 = GroupNorm(model_dim)

    def __call__(
        self, inputs, encoder_output, look_ahead_mask, padding_mask, training=True
    ):
        attn1, attn_weights_block1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        attn1 = tf.nn.dropout(attn1, rate=self.dropout_prob) if training else attn1
        out1 = self.groupnorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attn2 = tf.nn.dropout(attn2, rate=self.dropout_prob) if training else attn2
        out2 = self.groupnorm2(attn2 + out1)

        ffn_output = self.ff(out2)
        ffn_output = (
            tf.nn.dropout(ffn_output, rate=self.dropout_prob)
            if training
            else ffn_output
        )
        out3 = self.groupnorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2
