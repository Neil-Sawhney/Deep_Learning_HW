import tensorflow as tf

# MultiHeadAttention Layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout
        
        assert model_dim % num_heads == 0
        
        self.depth = model_dim // num_heads
        
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        
        self.wo = tf.keras.layers.Dense(model_dim)
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        depth = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, inputs):
        query, key, value, mask = inputs
        batch_size = tf.shape(query)[0]
        
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        output = self.wo(concat_attention)
        
        return output, attention_weights
