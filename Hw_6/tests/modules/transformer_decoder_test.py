import pytest


def test_autoregressive():
    import tensorflow as tf

    from helpers.adam import Adam
    from modules.transformer_decoder import TransformerDecoder

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    tf.random.set_seed(0x43966E87BD57227011B5B03B58785EC1)

    batch_size = 100
    context_length = 6
    num_heads = 8
    model_dim = 512
    ffn_dim = 2048
    num_blocks = 6

    input_text = "<SOS> Florida man bites dog. <EOS> Dog bites professor curro. <EOS>"

    transformer_decoder = TransformerDecoder(
        context_length,
        num_heads,
        model_dim,
        ffn_dim,
        num_blocks,
        input_text,
    )
    learning_rate = 0.0001
    adam = Adam(learning_rate)

    text, targets = transformer_decoder.get_tokens_and_targets()

    for _ in range(20):
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=text.shape[0], dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            input_tokens_batch = tf.gather(text, batch_indices)
            targets_batch = tf.gather(targets, batch_indices)

            labels = targets_batch
            logits = transformer_decoder(input_tokens_batch)
            current_train_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits,
                )
            )

        grads = tape.gradient(
            current_train_loss, transformer_decoder.trainable_variables
        )

        adam.apply_gradients(zip(grads, transformer_decoder.trainable_variables))

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                logits.numpy().argmax(axis=2).reshape(-1),
                labels.numpy().reshape(-1),
            ),
            tf.float32,
        )
    )

    assert accuracy == 1.0
    assert transformer_decoder.predict("<SOS> Florida") == " man bites dog."
    assert transformer_decoder.predict("Dog") == " bites professor curro."


if __name__ == "__main__":
    pytest.main([__file__])
