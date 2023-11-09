import tensorflow as tf


def tokenizer(text: tf.Tensor, num_word_to_tokenize: int):
    """
    Tokenize the input text.

    Args:
        text (tf.Tensor): The text to tokenize. Shape: [batch_size, text_length]
        num_word_to_tokenize (int): The number of words to tokenize.

    Returns:
        tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_word_to_tokenize]
    """

    tokens = tf.strings.split(text, sep=" ")
    tokens = tokens[:, :num_word_to_tokenize]

    tokens = tokens.to_tensor(default_value=b"<pad>")
    if tokens.shape[1] < num_word_to_tokenize:
        tokens = tf.pad(
            tokens,
            [
                [0, 0],
                [0, num_word_to_tokenize - tokens.shape[1]],
            ],
            constant_values=b"<pad>",
        )

    return tokens
