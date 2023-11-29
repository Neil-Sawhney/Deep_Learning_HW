import tensorflow as tf


class EmbedToVocabFile(tf.Module):
    def __init__(self, vocab_file, embedding_depth):
        rng = tf.random.get_global_generator()

        self.vocab_file = vocab_file
        vocab_initializer = tf.lookup.TextFileInitializer(
            vocab_file.name,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        )

        self.vocab_table = tf.lookup.StaticHashTable(
            vocab_initializer, default_value=-1
        )

        self.embedding_depth = embedding_depth
        stddev = tf.cast(
            tf.sqrt(2 / (self.vocab_table.size() * embedding_depth)), tf.float32
        )
        self.embedding = tf.Variable(
            rng.normal(
                shape=[self.vocab_table.size(), embedding_depth],
                stddev=stddev,
            ),
            trainable=True,
            name="Embedder/embedding",
        )

    def tokens_to_ids(self, tokens):
        """
        Convert the tokens into integers.

        Args:
            tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_tokenized_words]

        Returns:
            tf.Tensor: The integer IDs of the tokens.
        """
        return tf.cast(self.vocab_table.lookup(tokens), tf.int32)

    def decode(self, logits):
        """
        Decode the logits into tokens.

        Args:
            logits (tf.Tensor): The logits. Shape: [batch_size, num_tokenized_words, vocab_size]

        Returns:
            tf.Tensor: The tokens corresponding to the logits. Shape: [batch_size, num_tokenized_words]
        """
        reverse_vocab = []
        with open(self.vocab_file.name, "r", encoding="utf-8") as f:
            for line in f:
                reverse_vocab.append(line.strip())

        # Get the most likely token ID for each embedding
        probabilities = tf.nn.softmax(logits, axis=-1)

        # Get the most likely token ID for each embedding
        token_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int64)

        # Convert the token IDs into tokens
        tokens = tf.gather(reverse_vocab, token_ids)

        return tokens

    def get_vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
            int: The size of the vocabulary
        """
        return tf.cast(self.vocab_table.size(), tf.int32)

    def __call__(self, tokens):
        """
        Embed the tokenized text.

        Args:
            tokens (tf.Tensor): The tokenized text. Shape: [batch_size, num_tokenized_words]

        Returns:
            tf.Tensor: The embeddings of the tokens.
            Shape should be [batch_size, num_tokenized_words, embedding_depth]
        """
        token_ids = self.tokens_to_ids(tokens)
        embeddings = tf.nn.embedding_lookup(self.embedding, token_ids)
        return embeddings
