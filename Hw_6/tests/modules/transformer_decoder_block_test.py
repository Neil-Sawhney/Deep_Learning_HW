import pytest


@pytest.mark.parametrize(
    "batch_size, seq_len, model_dim, num_heads",
    [
        (1, 6, 12, 3),
        (5, 12, 36, 9),
        (10, 27, 108, 27),
    ],
)
def test_dimensionality(batch_size, seq_len, model_dim, num_heads):
    import tensorflow as tf

    from modules.transformer_decoder_block import TransformerDecoderBlock

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    ffn_dim = 2048

    decoder_block = TransformerDecoderBlock(num_heads, model_dim, ffn_dim)

    input_embeddings = tf.Variable(rng.normal(shape=[batch_size, seq_len, model_dim]))
    embeddings = decoder_block(input_embeddings)

    assert embeddings.shape == (batch_size, seq_len, model_dim)


if __name__ == "__main__":
    pytest.main([__file__])
