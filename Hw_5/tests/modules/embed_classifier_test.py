import pytest
import tensorflow as tf


def test_EmbedClassifier_init():
    from modules.embed_classifier import EmbedClassifier

    embed_classifier = EmbedClassifier(100, 50, 20, 0.5, 3, 30, 10)

    assert embed_classifier.num_embedding == 100
    assert embed_classifier.embedding_depth == 50
    assert embed_classifier.num_word_to_tokenize == 20
    assert embed_classifier.embedding.shape == (100, 50)
    assert embed_classifier.mlp is not None


@pytest.mark.parametrize(
    "text",
    [
        tf.constant(["professor curro likes", "reading about", "apples"]),
        tf.constant(["apples are a good fruit"]),
    ],
)
@pytest.mark.parametrize("num_embedding", [100, 200, 300])
@pytest.mark.parametrize("embedding_depth", [50, 100, 150])
@pytest.mark.parametrize("num_word_to_tokenize", [10, 20, 30])
@pytest.mark.parametrize("num_classes", [10, 20, 30])
def test_EmbedClassifier_call(
    text, num_embedding, embedding_depth, num_word_to_tokenize, num_classes
):
    from modules.embed_classifier import EmbedClassifier

    num_classes = 10
    embed_classifier = EmbedClassifier(
        num_embedding,
        embedding_depth,
        num_word_to_tokenize,
        dropout_prob=0.5,
        num_hidden_layers=3,
        hidden_layer_width=30,
        num_classes=num_classes,
    )

    embeddings = embed_classifier(text)

    assert embeddings.shape[0] == len(text)
    assert embeddings.shape[1] == num_classes


if __name__ == "__main__":
    pytest.main([__file__])
