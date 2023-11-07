import tensorflow as tf


class Embedder:
    def __init__(self, num_embedding, embedding_depth):
        rng = tf.random.get_global_generator()

        self.num_embedding = num_embedding
        self.embedding_depth = embedding_depth
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

    def __call__(self, tokens):
        """
        Embed the tokenized text.

        Args:
            tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_tokenized_words]

        Returns:
            tf.Tensor: The embeddings of the tokens.
            Shape should be [batch_size, num_word_to_tokenize * embedding_depth]
        """
        hashed_tokens = tf.strings.to_hash_bucket_fast(tokens, self.num_embedding)

        embeddings = tf.nn.embedding_lookup(self.embedding, hashed_tokens)
        num_tokenized_words = tokens.shape[1]
        embeddings = tf.reshape(
            embeddings, [-1, num_tokenized_words * self.embedding_depth]
        )

        return embeddings
