import tensorflow as tf

from helpers.embed_text import EmbedText
from helpers.positional_encoding import PositionalEncoding
from modules.linear import Linear
from modules.transformer_decoder_block import TransformerDecoderBlock


class TransformerDecoder(tf.Module):
    """Transformer Decoder.

    This is an implementation of a transformer decoder as described
    in the paper "Attention is all you Need" (Vaswani et al., 2017).

    Args:
        num_embedding: Number of embeddings to use.
        embedding_depth: Depth of each embedding.
        num_word_to_tokenize: Number of words to tokenize.
        num_heads: Number of attention heads.
        model_dim: Size of each attention head for value, query, and queue.
        ffn_dim: Size of the hidden layer in the feed-forward network.
        num_blocks: Number of transformer decoder blocks.
        dropout: Dropout probability.

    Call arguments:
        input: Input `Tensor` of shape `(B, seq_len)`.
        mask: Optional mask tensor of shape `(B, seq_len, seq_len)`.
    """

    def __init__(
        self,
        num_embedding,
        embedding_depth,
        num_word_to_tokenize,
        num_heads,
        model_dim,
        ffn_dim,
        num_blocks,
        dropout_prob=0.1,
    ):
        self.embed_text = EmbedText(
            num_embedding,
            embedding_depth,
            num_word_to_tokenize,
        )

        self.positional_encoding = PositionalEncoding(num_word_to_tokenize, model_dim)

        self.layers = [
            TransformerDecoderBlock(num_heads, model_dim, ffn_dim, dropout_prob)
            for _ in range(num_blocks)
        ]

        self.linear = Linear(model_dim, num_embedding)

    def __call__(self, input, mask=False, training=False):
        embeddings = self.embed_text(input)
        embeddings = self.positional_encoding(embeddings)

        for layer in self.layers:
            embeddings = layer(embeddings, mask, training)

        return self.linear(embeddings)