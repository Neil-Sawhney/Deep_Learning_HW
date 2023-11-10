import einops
import tensorflow as tf


class Tokenizer(tf.Module):
    def __init__(self, num_word_to_tokenize, pre_batched=True):
        self.num_word_to_tokenize = num_word_to_tokenize
        self.pre_batched = pre_batched

    def __call__(self, text: tf.Tensor):
        """
        Tokenize the input text.

        Args:
            text (tf.Tensor): The text to tokenize. Shape: [batch_size, text_length]

        Returns:
            tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_word_to_tokenize]
        """
        tokens = tf.strings.split(text, sep=" ")
        if self.pre_batched:
            tokens = tokens[:, : self.num_word_to_tokenize]
            tokens = tokens.to_tensor(default_value=b"<PAD>")
            if tokens.shape[1] < self.num_word_to_tokenize:
                tokens = tf.pad(
                    tokens,
                    [
                        [0, 0],
                        [0, self.num_word_to_tokenize - tokens.shape[1]],
                    ],
                    constant_values=b"<PAD>",
                )
        else:
            # Pad the sequence with <PAD> tokens to make it a multiple of context_length
            num_tokens = tokens.shape[0]
            remainder = num_tokens % (self.num_word_to_tokenize)

            if remainder != 0:
                tokens = tf.pad(
                    tokens, [[0, (self.num_word_to_tokenize) - remainder]]
                )

            tokens = einops.rearrange(
                tokens,
                "(batch context_length) -> batch context_length",
                context_length=self.num_word_to_tokenize,
            )

        return tokens
