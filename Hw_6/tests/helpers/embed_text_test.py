import pytest
import tensorflow as tf


def test_EmbedText_init():
    from helpers.embed_text import EmbedText

    embed_text = EmbedText(100, 50, 20)

    assert embed_text.num_embedding == 100
    assert embed_text.embedding_depth == 50
    assert embed_text.num_word_to_tokenize == 20
    assert embed_text.embedding.shape == (100, 50)


if __name__ == "__main__":
    pytest.main([__file__])
