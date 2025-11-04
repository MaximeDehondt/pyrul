# rulrve_core.py
# Architecture of the RULRVE model.
# Code mostly taken from https://github.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Masking
from tensorflow.keras.callbacks import LambdaCallback

class LRFinder:
    """
    Cyclical LR, code tailored from:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = self.model.optimizer.learning_rate.numpy()
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        self.model.optimizer.learning_rate.assign(lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1, **kw_fit):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = self.model.optimizer.learning_rate.numpy()

        # Set the initial learning rate
        self.model.optimizer.learning_rate.assign(start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=[callback],
                       verbose=0,
                       shuffle=True,
                       **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        self.model.optimizer.learning_rate.assign(original_lr)

    def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(generator)
            except (ValueError, NotImplementedError) as e:
                raise e('`steps_per_epoch=None` is only valid for a'
                        ' generator based on the '
                        '`keras.utils.Sequence`'
                        ' class. Please specify `steps_per_epoch` '
                        'or use the `keras.utils.Sequence` class.')
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs * steps_per_epoch))

        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = self.model.optimizer.learning_rate.numpy()

        # Set the initial learning rate
        self.model.optimizer.learning_rate.assign(start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch,
                                                      logs: self.on_batch_end(batch, logs))

        self.model.fit_generator(generator=generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[callback],
                                 **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        self.model.optimizer.learning_rate.assign(original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]
        
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, sigma) to sample z, the vector encoding an engine trajetory."""
    def __init__(self, seed):
        super(Sampling, self).__init__()
        self._seed = seed

    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=self._seed)
        return mu + tf.exp(0.5 * sigma) * epsilon

class EncoderLayer(tf.keras.Model):
    def __init__(self, 
                 timesteps: int, 
                 input_dim: int, 
                 masking_value: float, 
                 intermediate_dim: int, 
                 latent_dim: int,
                 seed: int):
        super(EncoderLayer, self).__init__()

        self.input_dim = input_dim
        self.masking_value = masking_value
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.mask = Masking(mask_value=masking_value)
        self.bilstm = Bidirectional(LSTM(intermediate_dim))

        # VAE Z layer
        self.mu = Dense(latent_dim)
        self.sigma = Dense(latent_dim)

        self.sampling = Sampling(seed)

    def call(self, inputs, training=False):
        x = self.mask(inputs)
        x = self.bilstm(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = self.sampling([mu, sigma])
        return [mu, sigma, z]

class RegressorLayer(tf.keras.Model):
    def __init__(self, latent_dim: int):
        super(RegressorLayer, self).__init__()

        self.latent_dim = latent_dim
        self.dense1 = Dense(200, activation='tanh')
        self.dense2 = Dense(1, name='reg_output')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
        
class RVE(tf.keras.Model):
    """
        RUL Variational Encoder, containing an encoder and a regressor layer.
    """
    def __init__(self, 
                 timesteps: int, 
                 input_dim: int, 
                 masking_value: float, 
                 intermediate_dim: int, 
                 latent_dim: int,
                 seed: int):
        super(RVE, self).__init__()
        self.encoder = EncoderLayer(timesteps, input_dim, masking_value, intermediate_dim, latent_dim, seed)
        self.regressor = RegressorLayer(latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
        
    def call(self, inputs):
        z = self.encoder(inputs)[-1]
        r = self.regressor(z)
        return r

    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "regressor": keras.saving.serialize_keras_object(self.regressor),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            self.reg_loss_tracker,
        ]

    def train_step(self, data):
        x, target_x = data

        with tf.GradientTape() as tape:
            # kl loss
            mu, sigma, z = self.encoder(x)
            kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # Regressor
            reg_prediction = self.regressor(z)
            reg_loss = tf.reduce_mean(
                keras.losses.mse(target_x, reg_prediction)
            )
            # Reconstruction
            total_loss = kl_loss + reg_loss
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "reg_loss": self.reg_loss_tracker.result(),
        }


    def test_step(self, data):
        x, target_x = data

        # kl loss
        mu, sigma, z = self.encoder(x)
        kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Regressor
        reg_prediction = self.regressor(z)
        reg_loss = tf.reduce_mean(
            keras.losses.mse(target_x, reg_prediction)
        )
        # Reconstruction
        total_loss = kl_loss + reg_loss

        return {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "reg_loss": reg_loss,
        }
