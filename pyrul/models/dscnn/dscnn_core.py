# dscnn_core.py
# Architecture of the DSCNN model.
import tensorflow as tf

class SE(tf.keras.layers.Layer):
    """
        Squeeze + Excitation Unit.
    """
    def __init__(self, nb_channels: int, reducRatio: int):
        super(SE, self).__init__()

        self.globalAvgPool = tf.keras.layers.GlobalAveragePooling2D()
        self.fcn1 = tf.keras.layers.Dense(units=int(nb_channels/reducRatio), activation="relu", kernel_initializer="he_normal")
        self.fcn2 = tf.keras.layers.Dense(units=nb_channels, activation="sigmoid", kernel_initializer="he_normal")
        self.multiply = tf.keras.layers.Multiply()

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=False):
        
        x = self.globalAvgPool(inputs)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.multiply([x,inputs]) # channel-wise multiplication
        return x

class SepConvBlock(tf.keras.layers.Layer):
    """
        Separable Convolutional block made of (Batch Norm + ReLU + Sep Conv)*2 + AvgPooling + SE Unit layers, with skip connection.
    """
    def __init__(self, nb_kernels: int, ps: int = 4, r: int = 16):
        super(SepConvBlock, self).__init__()
        self._nb_kernels = nb_kernels
        self._ps = ps
        self._r = r
        
        self.relu = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.sepconv1 = tf.keras.layers.SeparableConv2D(filters=nb_kernels,
                                            kernel_size=(8,1),
                                            padding='same',
                                            depthwise_initializer="he_normal",
                                            pointwise_initializer="he_normal")

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.sepconv2 = tf.keras.layers.SeparableConv2D(filters=nb_kernels,
                                            kernel_size=(8,1),
                                            padding='same',
                                            depthwise_initializer="he_normal",
                                            pointwise_initializer="he_normal")

        self.avgpool = tf.keras.layers.AvgPool2D(pool_size=(ps,1), strides=(ps,1), padding='same')
        self.se = SE(nb_channels=nb_kernels, reducRatio=r)
        
    def compute_output_shape(self, input_shape):
        remaining_size = max(input_shape[1]//self._ps, 1)
        return (input_shape[0], remaining_size, input_shape[2], self._nb_kernels)

    def build(self, input_shape):
        self._do_skip = input_shape[1:] == (1,1,self._nb_kernels) # If shape matches, we apply skip connection

    def call(self, inputs, training=False):
        x = self.sepconv1(self.relu(self.bn1(inputs, training=training)))
        x = self.sepconv2(self.relu(self.bn2(x, training=training)))
        x = self.avgpool(x)
        x = self.se(x)
        
        if self._do_skip:
            x += inputs
        return x

class SepConvNet(tf.keras.models.Model):
    """
        Deep separable convolutional network.
    """
    def __init__(self, nb_init_kernels=16, nb_sepConvBlocks=3):
        super().__init__()

        self.sepconv = tf.keras.layers.SeparableConv2D(filters=nb_init_kernels,
                                            kernel_size=(8,1),
                                            padding='same',
                                            depthwise_initializer="he_normal",
                                            pointwise_initializer="he_normal")
        self.relu = tf.keras.layers.ReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        self.avgPool = tf.keras.layers.AvgPool2D(pool_size=(4,1), strides=(4,1), padding='same')

        self.l_sepConvs = [SepConvBlock(nb_kernels=2**(i+1)*nb_init_kernels) for i in range(nb_sepConvBlocks)]
        self.globalAvgPool = tf.keras.layers.GlobalAveragePooling2D()
        self.fcn = tf.keras.layers.Dense(units=1, activation="relu", kernel_initializer="he_normal")

        self._loss = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = [tf.keras.metrics.Mean(name='mse')]

    def call(self, x, training=False):
        x = self.relu(self.bn(self.sepconv(x, training=training), training=training))
        x = self.avgPool(x)
        for l in self.l_sepConvs:
            x = l(x, training=training)
        x = self.globalAvgPool(x)
        x = self.fcn(x)
        return x

    def compute_loss(self, x, y, y_pred, sample_weight):
        loss = self._loss(y, y_pred)
        self.loss_tracker[0].update_state(loss)
        return loss

    @property
    def metrics(self):
        return self.loss_tracker