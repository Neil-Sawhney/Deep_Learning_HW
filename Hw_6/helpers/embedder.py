import tensorflow as tf


class Embedder(tf.Module):
    def __init__(self, num_embedding, embedding_depth, num_word_to_tokenize):
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
        num_tokenized_words = tokens.shape[1]
        embeddings = tf.reshape(
            embeddings, [-1, num_tokenized_words * self.embedding_depth]
        )

        return embeddings
