# gcutcn_core.py
# Architecture of the GCUTCN model.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback

class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    """
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())
        
    def get_lr(self):
        return self.model.optimizer.learning_rate.numpy()
        
    def get_momentum(self):
        return self.model.optimizer.beta_1
        
    def set_lr(self, lr):
        self.model.optimizer.learning_rate = lr
        
    def set_momentum(self, mom):
        self.model.optimizer.beta_1 = mom

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
        plt.show()

class AMFE(tf.keras.layers.Layer):
    """
        Deep part of Adaptive Multi-scale Feature Extractor, containing 3 dense layers each with dropout and LeakyRelu, and a residual connection.
    """
    def __init__(self, 
                 nb_neur1: int, 
                 nb_neur2: int, 
                 output_size: int,
                 dp1: float, 
                 dp2: float, 
                 dp3: float, 
                 lrelu: float
                 ):
        super(AMFE, self).__init__()
        self._output_size = output_size

        self.act = tf.keras.layers.LeakyReLU(lrelu)
        
        self.linear1 = tf.keras.layers.Dense(units=nb_neur1, activation="linear")
        self.dropout1 = tf.keras.layers.Dropout(dp1)
        
        self.linear2 = tf.keras.layers.Dense(units=nb_neur2, activation="linear")
        self.dropout2 = tf.keras.layers.Dropout(dp2)
        
        self.linear3 = tf.keras.layers.Dense(units=output_size, activation="linear")
        self.dropout3 = tf.keras.layers.Dropout(dp3)

        self.res_linear = tf.keras.layers.Dense(units=output_size, activation="linear")
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self._output_size)

    def call(self, inputs, training=False):
        x = inputs
        x = self.dropout1(self.act(self.linear1(x)), training=training)
        x = self.dropout2(self.act(self.linear2(x)), training=training)
        x = self.dropout3(self.act(self.linear3(x)), training=training)
        
        return x + self.res_linear(inputs)

class GCU(tf.keras.layers.Layer):
    """
        Gated Convolutional Unit, containing a CNN and a gating mechanism.
    """
    def __init__(self, ks: int, nb_features: int):
        super(GCU, self).__init__()

        self.conv = tf.keras.layers.Conv1D(filters=nb_features,
                                           kernel_size=ks,
                                           strides=1,
                                           padding="same"
                                          )
        self.denseU1 = tf.keras.layers.Dense(units=nb_features,
                                            activation="linear",
                                            use_bias=False)
        self.denseU2= tf.keras.layers.Dense(units=nb_features,
                                            activation="linear",
                                            use_bias=True)
        self.denseR1 = tf.keras.layers.Dense(units=nb_features,
                                            activation="linear",
                                            use_bias=False)
        self.denseR2 = tf.keras.layers.Dense(units=nb_features,
                                            activation="linear",
                                            use_bias=True)
        self.denseH = tf.keras.layers.Dense(units=nb_features,
                                            activation="linear",
                                            use_bias=True)
        
        self.sigmoid = tf.keras.activations.sigmoid

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=False):
        x = inputs
        h = self.conv(inputs)

        u = self.sigmoid(self.denseU1(h) + self.denseU2(x))
        r = self.sigmoid(self.denseR1(h) + self.denseR2(x))
        
        fusion = tf.math.multiply(h,u) + tf.math.multiply(x,r)
        res = self.denseH(fusion)
        return res

class TCN(tf.keras.layers.Layer):
    """
        Temporal Convolutional Unit, containing dilated causal convolution layers with weight normalization, dropout and a residual connection.
    """
    def __init__(self, 
                 nb_filters1: int, 
                 nb_filters2: int, 
                 ks: int, 
                 dp: int, 
                 dr1: int,
                 dr2: int,
                 lrelu: float
                 ):
        super(TCN, self).__init__()
        self._nb_filters2 = nb_filters2

        self.res_conv = tf.keras.layers.Conv1D(filters=nb_filters2,
                                            kernel_size=1,
                                            padding="same",
                                            kernel_initializer='he_normal'
                                           )
        self.conv1 = WeightNormCausalConv1D(filters=nb_filters1,
                                            kernel_size=ks,
                                            dilation_rate=dr1,
                                            kernel_initializer='he_normal'
                                           )
        self.dropout1 = tf.keras.layers.Dropout(dp)
        self.conv2 = WeightNormCausalConv1D(filters=nb_filters2,
                                            kernel_size=ks,
                                            dilation_rate=dr2,
                                            kernel_initializer='he_normal'
                                           )
        self.dropout2 = tf.keras.layers.Dropout(dp)
        self.act = tf.keras.layers.LeakyReLU(lrelu)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self._nb_filters2)

    def build(self, input_shape):
        self._do_skip_adjust = input_shape[-1] != self._nb_filters2 # If shapes don't match, we need a 1x1 conv to apply residual connection

    def call(self, inputs, training=False):
        x = self.dropout1(self.act(self.conv1(inputs)), training=training)
        x = self.dropout2(self.act(self.conv2(x)), training=training)
        if self._do_skip_adjust:
            x += self.res_conv(inputs)
        else:
            x += inputs
        return self.act(x)

class GCUTCN(tf.keras.models.Model):
    """
        GCU-TCN Model
    """
    def __init__(self, ):
        super().__init__()

        self.amfe = AMFE(nb_neur1=64, 
                         nb_neur2=128, 
                         output_size=32, 
                         dp1=0.1, 
                         dp2=0.1,
                         dp3=0.1, 
                         lrelu=0.1)
        self.gcu = GCU(ks=3, nb_features=32)
        self.tcn1 = TCN(nb_filters1=64, 
                        nb_filters2=32, 
                        ks=5, 
                        dp=0.1, 
                        dr1=1,
                        dr2=2,
                        lrelu=0.1)
        self.tcn2 = TCN(nb_filters1=16, 
                        nb_filters2=16, 
                        ks=5, 
                        dp=0.1, 
                        dr1=4,
                        dr2=6,
                        lrelu=0.1)
        self.tcn3 = TCN(nb_filters1=8, 
                        nb_filters2=8, 
                        ks=5, 
                        dp=0.1, 
                        dr1=8,
                        dr2=10,
                        lrelu=0.1)
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(units=1, activation="linear")
        
        self._loss = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = [tf.keras.metrics.Mean(name='mse')]

    def call(self, inputs, training=False):

        x = self.amfe(inputs, training=training)
        x = self.gcu(x, training=training)
        x = self.tcn1(x, training=training)
        x = self.tcn2(x, training=training)
        x = self.tcn3(x, training=training)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def compute_loss(self, x, y, y_pred, sample_weight):
        loss = self._loss(y, y_pred)
        self.loss_tracker[0].update_state(loss)
        return loss

    @property
    def metrics(self):
        return self.loss_tracker

class WeightNormCausalConv1D(tf.keras.layers.Layer):
    """
        Causal dilated Conv1D layer but with channel-wise weight normalization.
    """

    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, kernel_initializer: str):
        super(WeightNormCausalConv1D, self).__init__()

        self._bias = None
        self._filters = filters
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate
        self._kernel_initializer=kernel_initializer

    def build(self, input_shape=None):

        self._v = self.add_weight(
          name='v',
          shape=(self._kernel_size, input_shape[2], self._filters),
          initializer=self._kernel_initializer,
          dtype=tf.float32,
          trainable=True)
        
        self._g = self.add_weight(
          name='g',
          shape=(1,1,self._filters),
          initializer=self._kernel_initializer,
          dtype=tf.float32,
          trainable=True)

        self._bias = self.add_weight(
            name="bias",
            shape=(self._filters,),
            initializer='zeros',
            dtype=tf.float32,
            trainable=True)
        
        self.built = True
        
    def compute_output_shape(self, input_shape):
        output_length = int(np.floor(input_shape[1]-self._dilation_rate*(self._kernel_size-1)))
        return (input_shape[0], output_length, self._filters)

    def compute_causal_padding(self):
        left_pad = self._dilation_rate * (self._kernel_size - 1)
        causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        return causal_padding

    def call(self, inputs, training=False):
        # Apply causal padding to inputs.
        x = tf.keras.ops.pad(inputs, self.compute_causal_padding())

        kernel = tf.math.l2_normalize(self._v, axis=(0,1)) * self._g
        outputs = tf.keras.backend.conv1d(x=x, 
                                          kernel=kernel, 
                                          strides=[1], 
                                          padding="valid", 
                                          dilation_rate=self._dilation_rate,
                                          data_format="channels_last")

        rank = 1
        bias_shape = (1,) * (rank + 1) + (self._filters,)
        bias = tf.reshape(self._bias, bias_shape)
        outputs += bias
            
        return outputs