import pytest


def test_masked_multi_head_attention():
    import tensorflow as tf
    from tensorflow import linalg, ones, zeros
    from transformer import MultiHeadAttention

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_seq_length = 5  # Maximum length of the input sequence
    h = 1  # Number of self-attention heads
    d_k = 3  # Dimensionality of the linearly projected queries and keys
    d_v = 3  # Dimensionality of the linearly projected values
    d_model = 3  # Dimensionality of the model sub-layers' outputs
    batch_size = 1  # Batch size from the training process

    queries = rng.normal(shape=[batch_size, input_seq_length, d_k])
    keys = rng.normal(shape=[batch_size, input_seq_length, d_k])
    values = rng.normal(shape=[batch_size, input_seq_length, d_v])
    input = [queries, keys, values]
    """
     triangular matrix mask looks like this (it gets multiplied by 
     -1e9 to act as -inf as mentioned in paper)
      [[0., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1.],
       [0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0.]]
       """
    mask = 1 - linalg.band_part(ones((input_seq_length, input_seq_length)), -1, 0)

    attention = MultiHeadAttention(h, d_model)
    with tf.GradientTape() as tape:
        tape.watch(input)  # needed for non-Variables
        multihead_output = attention(input[0], input[1], input[2], mask)
    dy_dx = tape.jacobian(multihead_output, input)

    # dy_dx[2][0][0][0][0][0] not equal to zeroes -> first input token is
    #   dependant on start token
    # dy_dx[2][0][0][0][0][1] equal to zeroes -> first input token is not
    #   dependant on second token in sequence
    tf.debugging.assert_none_equal(
        dy_dx[2][0][0][0][0][0],
        zeros(shape=dy_dx[2][0][0][0][0][0].shape),
        summarize=2,
    )
    tf.debugging.assert_equal(
        dy_dx[2][0][0][0][0][1],
        zeros(shape=dy_dx[2][0][0][0][0][1].shape),
        summarize=2,
    )
    # to print out all 5 words print:
    # dy_dx[2][0][0][0][0][1] dy_dx[2][0][1][0][0][1]
    # dy_dx[2][0][2][0][0][1] dy_dx[2][0][3][0][0][1]
    # dy_dx[2][0][4][0][0][1]


if __name__ == "__main__":
    pytest.main([__file__])
