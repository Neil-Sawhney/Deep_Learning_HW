import tensorflow as tf


class EmbedText(tf.Module):
    """Embeds text into a sequence of embeddings.

    Args:
        num_embedding: Number of embeddings to use.
        embedding_depth: Depth of each embedding.
        num_word_to_tokenize: Number of words to tokenize.

    Call arguments:
        text: Text to tokenize.

    Returns:
        embeddings: Embeddings of the tokens. Shape should be
        [batch_size, num_word_to_tokenize * embedding_depth]
    """

    def __init__(
        self,
        num_embedding,
        embedding_depth,
        num_word_to_tokenize,
    ):
        rng = tf.random.get_global_generator()

        self.num_embedding = num_embedding
        self.embedding_depth = embedding_depth
        self.num_word_to_tokenize = num_word_to_tokenize

        stddev = tf.sqrt(2 / (self.num_embedding * self.embedding_depth))
        self.embedding = tf.Variable(
            rng.normal(
                shape=[
                    self.num_embedding,
                    self.embedding_depth,
                ],
                stddev=stddev,
            ),
            trainable=True,
            name="EmbedText/embedding",
        )

    def __call__(self, text: tf.Tensor):
        """Converts tokens to a flattened list of all embeddings

        Args:
            text (tf.Tensor): The text to tokenize.
            Shape should be [batch_size, num_word_to_tokenize]

        Returns:
            tf.Tensor: The embeddings of the tokens.
            Shape should be [batch_size, num_word_to_tokenize * embedding_depth]
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

        hashed_tokens = tf.strings.to_hash_bucket_fast(tokens, self.num_embedding)

        embeddings = tf.nn.embedding_lookup(self.embedding, hashed_tokens)
        embeddings = tf.reshape(
            embeddings, [-1, self.num_word_to_tokenize * self.embedding_depth]
        )

        return embeddings
