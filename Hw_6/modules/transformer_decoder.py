import tempfile

import tensorflow as tf

from helpers.embed_to_vocab_file import EmbedToVocabFile
from helpers.positional_encoding import PositionalEncoding
from helpers.tokenizer import Tokenizer
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
        input_text: Text to use for training. If None, training will not be possible.
        vocab_file: Path to the vocab file. If None, a temporary file will be created.
        dropout: Dropout probability.

    Call arguments:
        input: Input `Tensor` of shape `(B, seq_len)`.

    Returns:
        Output `Tensor` of shape `(B, seq_len, vocab_size)`.
    """

    def __init__(
        self,
        context_length,
        num_heads,
        model_dim,
        ffn_dim,
        num_blocks,
        input_text=None,
        vocab_file=None,
        dropout_prob=0.1,
    ):
        self.input_text = input_text
        self.context_length = context_length
        if vocab_file is None:
            vocab_file = self.create_vocab_file(input_text)

        self.embedder = EmbedToVocabFile(vocab_file, model_dim)
        self.positional_encoding = PositionalEncoding(context_length, model_dim)

        self.layers = [
            TransformerDecoderBlock(num_heads, model_dim, ffn_dim, dropout_prob)
            for _ in range(num_blocks)
        ]

        self.vocab_size = self.embedder.get_vocab_size()
        self.linear = Linear(model_dim, self.vocab_size)

    def create_vocab_file(self, input_text):
        # Tokenize the contents of the file
        tokenizer = Tokenizer(self.context_length, False)
        tokenized_text = tokenizer(input_text)

        # Flatten the tokenized_text tensor to 1D
        flattened_text = tf.reshape(tokenized_text, [-1])

        # Create a tensor of unique tokens
        unique_tokens, _ = tf.unique(flattened_text)

        # Write these unique tokens to a new vocab file
        vocab_file = tempfile.NamedTemporaryFile(delete=False)
        with open(vocab_file.name, "w", encoding="utf-8") as vocab_file:
            for token in unique_tokens.numpy():
                vocab_file.write(f"{token.decode('utf-8')}")
                if token != unique_tokens[-1]:
                    vocab_file.write("\n")
        return vocab_file

    def get_tokens_and_targets(self):
        tokenizer = Tokenizer(self.context_length, False)
        tokenized_text = tokenizer(self.input_text)

        tokenized_targets = tokenized_text[:, 1:]
        tokenized_text = tokenized_text[:, :-1]

        targets = self.embedder.tokens_to_ids(tokenized_targets)

        return tokenized_text, targets

    def decode(self, logits):
        return self.embedder.decode(logits)

    def __call__(self, input_tokens, training=True):
        embeddings = self.embedder(input_tokens)
        embeddings = self.positional_encoding(embeddings)

        mask = tf.linalg.band_part(
            tf.ones((input_tokens.shape[1], input_tokens.shape[1])), 0, -1
        )

        # make the main diagonal 0
        mask = mask - tf.eye(input_tokens.shape[1])

        for layer in self.layers:
            embeddings = layer(embeddings, mask, training)

        output = self.linear(embeddings)
        return output
