# dast_core.py
# Architecture of the DAST model.
# Code mostly inspired from : https://github.com/Zzzsdu/DAST
import tensorflow as tf
import numpy as np

def att_norm(Q, K):
    m = tf.matmul(Q, tf.transpose(K,perm=[0,2,1])) / Q.shape[-1]**0.5
    return tf.nn.softmax(m,-1)

def attention(Q, K, V):
    a = att_norm(Q, K)
    return  tf.matmul(a, V)

class Embedding(tf.keras.layers.Layer):
    """
        Embedding layer.
    """
    def __init__(self, dim_val):
        super(Embedding, self).__init__()
        self.dim_val = dim_val    
        self.fc1 = tf.keras.layers.Dense(units=dim_val, use_bias=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim_val)
    
    def call(self, x, training=False):
        x = self.fc1(x)       
        return x

class AttentionBlock(tf.keras.layers.Layer):
    """
        Attention block.
    """
    def __init__(self, 
                 dim_val: int, 
                 dim_attn: int
                 ):
        super(AttentionBlock, self).__init__()
        self.dim_val = dim_val
        self.value = Embedding(dim_val)
        self.key = Embedding(dim_attn)
        self.query = Embedding(dim_attn)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim_val)
    
    def call(self, x, kv=None, training=False): 
        if (kv is None):
            return attention(self.query(x), self.key(x), self.value(x))        
        return attention(self.query(x), self.key(kv), self.value(kv))

class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    """
        Multi-head self-attention block.
    """
    def __init__(self, 
                 dim_val: int, 
                 dim_attn: int, 
                 n_heads: int
                 ): 
        super(MultiHeadAttentionBlock, self).__init__()
        self.dim_val = dim_val
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))       
        #self.heads = nn.ModuleList(self.heads)
        self.fc = tf.keras.layers.Dense(units=dim_val, use_bias=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim_val)
        
    def call(self, x, kv=None, training=False):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
        a = tf.stack(a, axis=-1)
        a = tf.reshape(a, [tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2]*tf.shape(a)[3]])
        x = self.fc(a)
        return x

class PositionalEncoding(tf.keras.layers.Layer):
    """
        Positional Encoding layer.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float=0.1, 
                 max_len: int=512
                 ):
        super(PositionalEncoding, self).__init__()

        pe = np.zeros((max_len, d_model)) # (max_len, d_model)
        position = np.expand_dims(np.arange(0, max_len, dtype="float32"),-1) # (max_len)
        div_term = np.exp(np.arange(0, d_model, 2, dtype="float32") * (-np.log(10000.0) / d_model)) # (d_model//2)
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x, training=False):
        x = x + self.pe[:x.shape[1], :]
        return x


class EncoderLayer(tf.keras.layers.Layer):
    """
        Encoder layer, made of : multi-head self-attention, add+norm, dense, add+norm layers
    """
    def __init__(self, 
                 dim_val: int, 
                 dim_attn: int, 
                 n_heads: int=1,
                 dropout: float=None
                 ):
        super(EncoderLayer, self).__init__()
        self.dim_val = dim_val

        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = tf.keras.layers.Dense(units=dim_val)
        self.fc2 = tf.keras.layers.Dense(units=dim_val)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        #self.dropout = tf.keras.layers.Dropout(dropout)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim_val)

    def call(self, x, training=False):
        a = self.attn(x)
        x = self.norm1(x + a)
        a = self.fc1(tf.keras.activations.elu(self.fc2(x)))
        x = self.norm2(x + a)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    """
        Decoder layer, made of : multi-head self-attention, add+norm, multi-head self-attention, add+norm, dense, add+norm layers.
    """
    def __init__(self, 
                 dim_val: int, 
                 dim_attn: int, 
                 n_heads: int=1, 
                 dropout: float=None
                 ):
        super(DecoderLayer, self).__init__()
        self.dim_val = dim_val

        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = tf.keras.layers.Dense(units=dim_val)
        self.fc2 = tf.keras.layers.Dense(units=dim_val)
        #self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim_val)

    def call(self, x, enc=None, training=False):
        a = self.attn1(x)
        x = self.norm1(a + x)
        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)
        a = self.fc1(tf.keras.activations.elu(self.fc2(x)))
        x = self.norm3(x + a)
        return x



class DASTNet(tf.keras.models.Model):
    """
        Dual Aspect Self-attention based Transformer.
    """
    def __init__(self, 
                 dim_val_s: int = 128, 
                 dim_attn_s: int = 128, 
                 dim_val_t: int = 128,
                 dim_attn_t: int = 128,
                 dim_val: int = 128, 
                 dim_attn: int = 128,
                 dec_seq_len: int = 4, 
                 out_seq_len: int = 1, 
                 n_decoder_layers: int = 1, 
                 n_encoder_layers: int = 1,
                 n_heads: int = 1, 
                 dropout: float = None):
        super().__init__()

        self.dec_seq_len = dec_seq_len
        #self._dropout = tf.keras.layers.Dropout(dropout) # UNUSED

        # Initiate Sensors, Time_step encoders and decoder
        self.sensor_encoder = [EncoderLayer(dim_val_s, dim_attn_s, n_heads) for _ in range(n_encoder_layers)]
        self.time_encoder = [EncoderLayer(dim_val_t, dim_attn_t, n_heads) for _ in range(n_encoder_layers)]
        self.decoder = [DecoderLayer(dim_val, dim_attn, n_heads) for _ in range(n_decoder_layers)]        

        #self.pos_s = PositionalEncoding(dim_val_s)
        self.pos_t = PositionalEncoding(dim_val_t)
        self.timestep_enc_input_fc = tf.keras.layers.Dense(units=dim_val_t)
        self.sensor_enc_input_fc = tf.keras.layers.Dense(units=dim_val_s)
        
        self.dec_input_fc = tf.keras.layers.Dense(units=dim_val)
        self.out_fc = tf.keras.layers.Dense(units=out_seq_len)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.flat = tf.keras.layers.Flatten()

        self._loss = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = [tf.keras.metrics.Mean(name='mse')]

    def call(self, x, training=False):

        # Input embedding and positional encoding
        sensor_x = tf.transpose(x,perm=[0,2,1])
       
        e = self.sensor_encoder[0](self.sensor_enc_input_fc(sensor_x)) # (batch_size,sensor,dim_val_s)
        o = self.time_encoder[0](self.pos_t(self.timestep_enc_input_fc(x))) # (batch_size,timestep,dim_val_t)

        # Sensors encoder
        if len(self.sensor_encoder)>1:
            for sensor_enc in self.sensor_encoder[1:]:
                e = sensor_enc(e)
        
        # Time step encoder
        if len(self.time_encoder)>1:
            for time_enc in self.time_encoder[1:]:
                o = time_enc(o)
            
        # Feature fusion
        p = tf.keras.layers.Concatenate(axis = 1)([e,o])  # (batch_size,sensor+timestep,dim_val)
        p = self.norm1(p)
        
        # Decoder receive the output of feature fusion layer.        
        d = self.decoder[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), enc=p)
        
        if len(self.decoder)>1: # Decoder layers
            for dec in self.decoder[1:]:
                d = dec(d)
            
        # Output the RUL
        x = self.out_fc(tf.keras.activations.elu(self.flat(d)))
        return x

    def compute_loss(self, x, y, y_pred, sample_weight):
        loss = self._loss(y, y_pred)
        self.loss_tracker[0].update_state(loss)
        return loss

    @property
    def metrics(self):
        return self.loss_tracker

