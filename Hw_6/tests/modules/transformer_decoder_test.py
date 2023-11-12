import pytest


def test_causal_mask():
    import tensorflow as tf

    from modules.transformer_decoder import TransformerDecoder

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    context_length = 5
    num_heads = 8
    model_dim = 512
    ffn_dim = 2048
    num_blocks = 6
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

    # ensure that the derivative is zero for future tokens with respect to previous tokens
    with tf.GradientTape() as tape:
        logits = transformer_decoder(tokenized_text)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )
        )

    embed_variables = transformer_decoder.trainable_variables[0]

    gradients = tape.gradient(loss, embed_variables)

    


if __name__ == "__main__":
    pytest.main([__file__])
