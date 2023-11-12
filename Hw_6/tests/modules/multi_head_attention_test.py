import pytest


def test_masked_multi_head_attention():
    import tensorflow as tf

    from modules.multi_head_attention import MultiHeadAttention

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    context_length = 5
    num_heads = 1
    model_dim = 3
    batch_size = 1

    queries = rng.normal(shape=[batch_size, context_length, model_dim])
    keys = rng.normal(shape=[batch_size, context_length, model_dim])
    values = rng.normal(shape=[batch_size, context_length, model_dim])

    mask = tf.linalg.band_part(tf.ones((context_length, context_length)), 0, -1)
    # make the main diagonal 0
    mask = mask - tf.eye(context_length)

    # stack causal mask for each batch resulting in shape (B, seq_len, seq_len)
    mask = tf.stack([mask for _ in range(batch_size)])

    mha = MultiHeadAttention(num_heads, model_dim)

    with tf.GradientTape() as tape:
        tape.watch([queries, keys, values])
        output = mha(queries, keys, values, mask)

    dy_dx = tape.gradient(output, [queries, keys, values])

    # ensure that the derivative is zero for future tokens with respect to previous tokens
    assert tf.reduce_all(dy_dx[0][:, 0, 1:] == 0)

    # ensure that the derivative is not zero for future tokens with respect to previous tokens
    assert tf.reduce_all([dy_dx[0][:, i] != 0 for i in range(1, context_length)])


if __name__ == "__main__":
    pytest.main([__file__])
