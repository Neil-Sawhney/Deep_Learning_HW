import tensorflow as tf


def tokenizer(text: tf.Tensor):
    """
    Tokenize the input text.

    Args:
        text (tf.Tensor): The text to tokenize. Shape: [batch_size, text_length]

    Returns:
        tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_word_to_tokenize]
    """
    tokens = tf.strings.split(text, sep=" ")

    return tokens
