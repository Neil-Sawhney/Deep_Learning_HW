import pytest
import tensorflow as tf
from modules.embed_classifier import EmbedClassifier


def test_EmbedClassifier_init():
    embed_classifier = EmbedClassifier(100, 50, 20, 0.5, 3, 30, 10)

    assert embed_classifier.num_embedding == 100
    assert embed_classifier.embedding_depth == 50
    assert embed_classifier.num_word_to_tokenize == 20
    assert embed_classifier.embedding.shape == (100, 50)
    assert embed_classifier.mlp is not None


@pytest.mark.parametrize(
    "text",
    [
        tf.constant(["this is a test", "another test"]),
        tf.constant(["single test"]),
    ],
)
def test_EmbedClassifier_call(text):
    embed_classifier = EmbedClassifier(100, 50, 20, 0.5, 3, 30, 10)
    embeddings = embed_classifier(text)

    assert embeddings.shape[0] == len(text)
    assert (
        embeddings.shape[1]
        == embed_classifier.num_word_to_tokenize * embed_classifier.embedding_depth
    )


def test_EmbedClassifier_parameters_trainability():
    embed_classifier = EmbedClassifier(100, 50, 20, 0.5, 3, 30, 10)

    assert embed_classifier.embedding.trainable == True
    for layer in embed_classifier.mlp.layers:
        assert layer.trainable == True


if __name__ == "__main__":
    pytest.main([__file__])
