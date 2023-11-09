import tensorflow as tf

from helpers.embedder import Embedder
from helpers.tokenizer import Tokenizer
from modules.mlp import MLP


class EmbedClassifier(tf.Module):
    def __init__(
        self,
        num_embedding,
        embedding_depth,
        num_word_to_tokenize,
        dropout_prob,
        num_hidden_layers,
        hidden_layer_width,
        num_classes,
    ):
        self.tokenizer = Tokenizer(num_word_to_tokenize)
        self.embedder = Embedder(num_embedding, embedding_depth)

        self.mlp = MLP(
            num_word_to_tokenize * embedding_depth,
            num_classes,
            num_hidden_layers,
            hidden_layer_width,
            tf.nn.relu,
            tf.nn.softmax,
            dropout_prob,
        )

    def __call__(self, text: tf.Tensor):
        """Applies the embedding and MLP to the text.

        Args:
            text (tf.Tensor): The text to tokenize.
            Shape should be [batch_size]

        Returns:
            tf.Tensor: The logits of the classes.
            Shape should be [batch_size, num_classes]
        """

        tokens = self.tokenizer(text)
        embeddings = self.embedder(tokens)

        return self.mlp(embeddings)
