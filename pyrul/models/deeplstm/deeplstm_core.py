# deeplstm_core.py
# Architecture of the DeepLSTM model.
import tensorflow as tf

class DeepLSTM(tf.keras.models.Model):
    """
        Deep Long Short Term Memory network.
    """
    def __init__(self, 
                 lstm1_n: int, 
                 lstm2_n: int, 
                 fnn1_n: int, 
                 fnn2_n: int, 
                 lstm1_dp: float, 
                 lstm2_dp: float, 
                 fnn1_dp: float, 
                 fnn2_dp: float, 
                 l2_pen: float):
        super().__init__()
        # For efficiency and as is common, the activation functions are chosen as relu for dense layer, with He normal initialization.
        self.mask = tf.keras.layers.Masking(mask_value=-1.)
        
        self.lstm1 = tf.keras.layers.LSTM(units=lstm1_n, dropout=fnn1_dp, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l2=l2_pen))
        self.lstm2 = tf.keras.layers.LSTM(units=lstm2_n, dropout=fnn2_dp, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l2=l2_pen))
        self.fnn1 = tf.keras.layers.Dense(units=fnn1_n, activation="relu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(l2=l2_pen))
        self.dropout1 = tf.keras.layers.Dropout(fnn1_dp)
        self.fnn2 = tf.keras.layers.Dense(units=fnn2_n, activation="relu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(l2=l2_pen))
        self.dropout2 = tf.keras.layers.Dropout(fnn2_dp)
        self.final_fnn = tf.keras.layers.Dense(units=1, activation="linear", kernel_regularizer=tf.keras.regularizers.L2(l2=l2_pen))
        
        self._loss = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = [tf.keras.metrics.Mean(name='mse')]

    def call(self, 
             inputs, 
             training=False):

        x = self.mask(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.fnn1(x)
        x = self.dropout1(x, training=training)
        x = self.fnn2(x)
        x = self.dropout2(x, training=training)
        x = self.final_fnn(x)
        return x

    def compute_loss(self, 
                     x, 
                     y, 
                     y_pred, 
                     sample_weight):

        # mask is transmitted to this loss
        loss = self._loss(y, y_pred)      
        self.loss_tracker[0].update_state(loss)
        return loss

    @property
    def metrics(self):
        return self.loss_tracker