# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.calculate_positional_encoding(max_position, d_model)
        
    def calculate_positional_encoding(self, max_position, d_model):
        positional_encoding = np.zeros((max_position, d_model))
        position = np.arange(0, max_position, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.convert_to_tensor(positional_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Decoder Layer
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)
        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, enc_output, look_ahead_mask, padding_mask, training=True):
        attn1, attn_weights_block1 = self.mha1([inputs, inputs, inputs, look_ahead_mask])
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + inputs)
        
        attn2, attn_weights_block2 = self.mha2([out1, enc_output, enc_output, padding_mask])
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.feedforward(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

# Transformer Decoder
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, dff, target_vocab_size,
                 max_position_encoding, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_position_encoding, d_model)
        self.dec_layers = [TransformerDecoderLayer(num_heads, d_model, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, enc_output, look_ahead_mask, padding_mask, training=True):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        
        # Embedding and positional encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.embedding.output_dim, tf.float32))
        x = self.positional_encoding(x)
        
        # Decoder layers
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer_{i+1}_block1'] = block1
            attention_weights[f'decoder_layer_{i+1}_block2'] = block2
            
        return x, attention_weights