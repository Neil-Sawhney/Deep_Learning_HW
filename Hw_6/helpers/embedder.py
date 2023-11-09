import tensorflow as tf


class Embedder(tf.Module):
    def __init__(self, embedding_buckets, embedding_depth):
        rng = tf.random.get_global_generator()

        self.embedding_buckets = embedding_buckets
        self.embedding_depth = embedding_depth
        stddev = tf.sqrt(2 / (embedding_buckets * embedding_depth))
        self.embedding = tf.Variable(
            rng.normal(
                shape=[
                    embedding_buckets,
                    embedding_depth,
                ],
                stddev=stddev,
            ),
            trainable=True,
            name="Embedder/embedding",
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
        hashed_tokens = tf.strings.to_hash_bucket_fast(tokens, self.embedding_buckets)

        embeddings = tf.nn.embedding_lookup(self.embedding, hashed_tokens)

        return embeddings
