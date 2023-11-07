import tensorflow as tf


class Tokenizer(tf.Module):
    def __init__(self, num_word_to_tokenize):
        self.num_word_to_tokenize = num_word_to_tokenize

    def __call__(self, text: tf.Tensor):
        """
        Tokenize the input text.

        Args:
            text (tf.Tensor): The text to tokenize. Shape: [batch_size, text_length]

        Returns:
            tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_word_to_tokenize]
        """
        tokens = tf.strings.split(text, sep=" ")
        tokens = tokens[:, : self.num_word_to_tokenize]
        tokens = tokens.to_tensor(default_value="")
        if tokens.shape[1] < self.num_word_to_tokenize:
            tokens = tf.pad(
                tokens,
                [
                    [0, 0],
                    [0, self.num_word_to_tokenize - tokens.shape[1]],
                ],
            )

        return tokens
