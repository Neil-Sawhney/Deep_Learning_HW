import pytest


@pytest.mark.parametrize(
    "sentence, expected_tokens",
    [
        (
            "Woah oh wee woo Wah wi wah woo",
            [[b"Woah", b"oh", b"wee", b"woo", b"Wah", b"wi", b"wah", b"woo"]],
        ),
        (
            "AHHHHHHH AHHHHHH AHH AHHHHHHH AHHHHHHHHHHHHHH AHH AHAHHH AHHH AHHH AHH",
            [
                [
                    b"AHHHHHHH",
                    b"AHHHHHH",
                    b"AHH",
                    b"AHHHHHHH",
                    b"AHHHHHHHHHHHHHH",
                    b"AHH",
                    b"AHAHHH",
                    b"AHHH",
                ],
                [
                    b"AHHH",
                    b"AHH",
                    b"<PAD>",
                    b"<PAD>",
                    b"<PAD>",
                    b"<PAD>",
                    b"<PAD>",
                    b"<PAD>",
                ],
            ],
        ),
    ],
)
def test_tokenizer(sentence, expected_tokens):
    import tensorflow as tf

    from helpers.tokenizer import Tokenizer

    tokenizer = Tokenizer(8, False)

    tokens = tokenizer(sentence)

    assert tf.reduce_all(tf.equal(tokens, tf.convert_to_tensor(expected_tokens)))


if __name__ == "__main__":
    pytest.main([__file__])
