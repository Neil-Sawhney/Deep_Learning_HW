import pytest


def test_causal_mask():
    import tensorflow as tf

    from modules.transformer_decoder import TransformerDecoder

    context_length = 5
    num_heads = 10
    model_dim = 10
    ffn_dim = 10
    num_blocks = 10
    input_text = "<SOS> Man Bites Dog <EOS>"

    transformer_decoder = TransformerDecoder(
        context_length,
        num_heads,
        model_dim,
        ffn_dim,
        num_blocks,
        input_text,
    )

    tokenized_text, targets = transformer_decoder.get_tokens_and_targets()
    logits = transformer_decoder(tokenized_text)

    # ensure that the derivative is zero for future tokens with respect to previous tokens
    with tf.GradientTape() as tape:
        tape.watch(transformer_decoder.embeddings)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )
        )
    grads = tape.gradient(loss, transformer_decoder.embeddings)
    assert tf.reduce_all(grads[:, 0, :] == 0)


if __name__ == "__main__":
    pytest.main([__file__])
