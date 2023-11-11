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
        input: Input `Tensor` of shape `(B, seq_len)` during training and untokenized `(B, 1)` during inference.

    Returns:
        Output `Tensor` of shape `(B, seq_len, vocab_size)` during training and untokenized `(B, 1)` during inference.
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
        self.tokenizer = Tokenizer(context_length, False)

        self.input_text = input_text
        self.context_length = context_length
        if vocab_file is None:
            self.vocab_file = self._create_vocab_file(input_text)
        else:
            self.vocab_file = tf.io.gfile.GFile(vocab_file, "r")

        self.embedder = EmbedToVocabFile(self.vocab_file, model_dim)
        self.positional_encoding = PositionalEncoding(context_length, model_dim)

        self.layers = [
            TransformerDecoderBlock(num_heads, model_dim, ffn_dim, dropout_prob)
            for _ in range(num_blocks)
        ]

        self.vocab_size = self.embedder.get_vocab_size()
        self.linear = Linear(model_dim, self.vocab_size)

    def get_vocab_file(self):
        return self.vocab_file

    def get_tokens_and_targets(self):
        tokenized_text = self.tokenizer(self.input_text)

        tokenized_targets = tokenized_text[:, 1:]
        tokenized_text = tokenized_text[:, :-1]

        targets = self.embedder.tokens_to_ids(tokenized_targets)

        return tokenized_text, targets

    def decode(self, logits):
        return self.embedder.decode(logits)

    def predict(self, input_text):
        tokenized_text = self.tokenizer(input_text)

        len_input = len(input_text.split())

        output_index = len_input
        output = ""
        for _ in range(self.context_length - len_input):
            logits = self.__call__(tokenized_text, training=False)

            decoded_logits = self.decode(logits)

            next_word = decoded_logits[:, output_index - 1 : output_index]

            next_word_decoded = next_word[-1][0].numpy().decode("utf-8")

            output_index += 1

            if next_word_decoded == "<EOS>":
                break

            elif next_word_decoded == "<PAD>":
                continue

            output += " " + next_word_decoded

        return output

    def _create_vocab_file(self, input_text):
        # Tokenize the contents of the file
        tokenized_text = self.tokenizer(input_text)

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

    def __call__(self, input_tokens, training=True):
        causal_mask = tf.linalg.band_part(
            tf.ones((input_tokens.shape[1], input_tokens.shape[1])), 0, -1
        )
        # make the main diagonal 0
        causal_mask = causal_mask - tf.eye(input_tokens.shape[1])

        # stack causal mask for each batch resulting in shape (B, seq_len, seq_len)
        causal_mask = tf.stack([causal_mask for _ in range(input_tokens.shape[0])])

        pad_mask_vector = tf.cast(tf.equal(input_tokens, b"<PAD>"), tf.float32)

        # stack seq_len copies of the pad mask vector for each batch resulting in shape (B, seq_len, seq_len)
        pad_mask1 = tf.stack(
            [pad_mask_vector for _ in range(input_tokens.shape[1])], axis=1
        )

        # transpose the pad mask vector to get shape (B, seq_len, seq_len)
        pad_mask2 = tf.transpose(pad_mask1, [0, 2, 1])

        # logical or of the two pad masks, switch 2's to 1's
        pad_mask = tf.cast(pad_mask1 + pad_mask2, tf.bool)
        pad_mask = tf.cast(pad_mask, tf.float32)
        pad_mask = pad_mask1

        mask = tf.cast(causal_mask + pad_mask, tf.bool)
        mask = tf.cast(mask, tf.float32)

        embeddings = self.embedder(input_tokens)

        for layer in self.layers:
            embeddings = layer(embeddings, mask, training)

        output = self.linear(embeddings)

        return output
