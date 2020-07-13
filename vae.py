from utils import dataloader, visualizer

import os
import matplotlib.pyplot as plt
import numpy as np
import gc

from keras.models import Model, load_model
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, LSTM, Bidirectional, Dense, Reshape
from keras.layers import Flatten, Lambda
from keras.layers import UpSampling2D, Conv2DTranspose

from keras.losses import binary_crossentropy, mean_squared_error
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras import backend as K

class AEOD:

    def __init__(self,
                 # Data loading options
                 dataset='extended',
                 norm_method='min_max_per_core',
                 val_fold=1,
                 crop=None,
                 inv_thresh=0.4,
                 custom_data='benign',
                 # Network setup
                 load_model_path = None,
                 n_filters=128,
                 kernel_size=16,
                 strides=1,
                 pool_size=4,
                 n_units=[64, 32, 16],
                 latent_dim=10,
                 # Compiler options
                 loss=['mse', 'mse'],
                 optimizer='adam',
                 # Training options
                 epochs=150,
                 batch_size=256):

        # Arguments
        self.dataset = dataset
        self.norm_method = norm_method
        self.val_fold = val_fold
        self.crop = crop
        self.inv_thresh = inv_thresh
        self.custom_data = custom_data
        self.load_model_path = load_model_path
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_dim = latent_dim
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        # Initialize
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.history = None
        self.dl = None

    def visualize(self, num_examples=5, hist_range=None, del_data=True):

        if (del_data is True):
            del self.dl

        vs = visualizer(model=self.autoencoder,
                        history=self.history,
                        dataset=self.dataset,
                        norm_method=self.norm_method,
                        val_fold=self.val_fold,
                        crop=self.crop,
                        inv_thresh=self.inv_thresh)

        vs.training_curve()
        vs.benign_cancer_examples(num_examples)
        vs.error_distribution(hist_range)

    def fit(self):

        # Prepare data
        self.dl = dataloader(dataset=self.dataset,
                             norm_method=self.norm_method,
                             val_fold=self.val_fold,
                             crop=self.crop,
                             inv_thresh=self.inv_thresh,
                             custom_data=self.custom_data,
                             format='Squashed',
                             verbose=2)

        # Init network
        self.input_dim = 1
        self.timesteps=int(self.dl.data_train.shape[1])
        self.build_ae()

        # Prepare callbacks
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.000)
        filepath = "model-ae-epoch-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=False, period=10)

        # Fit model
        self.history = self.autoencoder.fit(x=self.dl.data_train,
                                            y=self.dl.data_train,
                                            validation_data=[self.dl.data_val, self.dl.data_val],
                                            nb_epoch=self.epochs,
                                            batch_size=self.batch_size,
                                            verbose=1,
                                            callbacks=[early_stop, checkpoint]).history

    def predict(self, x):
        return self.autoencoder.predict(x)

    def build_ae(self):

        if (self.load_model_path is None):

            assert(self.timesteps % self.pool_size == 0)

            # Encoder
            x1 = Input(shape=(self.timesteps, self.input_dim), name='encoder_input')
            encoded = Conv1D(self.n_filters, self.kernel_size, strides=self.strides, padding='same', activation='linear')(x1)
            encoded = LeakyReLU()(encoded)
            encoded = MaxPool1D(self.pool_size)(encoded)

            for i in range(len(self.n_units)):
                encoded = Bidirectional(LSTM(self.n_units[i], return_sequences=True), merge_mode='sum')(encoded)
                encoded = LeakyReLU()(encoded)

            encoded_shape = K.int_shape(encoded)
            encoded_output = Flatten()(encoded)

            # Create encoder model
            self.encoder = Model(inputs=x1, outputs=encoded_output, name='encoder')
            self.encoder.summary()

            # Decoder
            latent_input = Input(shape=(self.tuple_product(encoded_shape),), name='z_sampling')
            decoded = Reshape(encoded_shape[1:])(latent_input)

            for i in range(len(self.n_units)):
                decoded = Bidirectional(LSTM(self.n_units[len(self.n_units)-1-i], return_sequences=True), merge_mode='sum')(decoded)
                decoded = LeakyReLU()(decoded)

            decoded = Reshape((-1, 1, self.n_units[0]), name='reshape')(decoded)
            decoded = UpSampling2D((self.pool_size, 1), name='upsampling')(decoded)
            decoded = Conv2DTranspose(self.input_dim, (self.kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
            decoder_output = Reshape((-1, self.input_dim), name='output_seq')(decoded)

            # Create decoder model
            self.decoder = Model(inputs=latent_input, outputs=decoder_output, name='decoder')
            self.decoder.summary()

            # Connect components
            l1 = self.encoder(x1)
            x2 = self.decoder(l1)
            l2 = self.encoder(x2)

            # Create VAE model
            self.autoencoder = Model(inputs=x1, outputs=x2, name='autoencoder')

            # self.autoencoder.load_weights('./models/model-ae-epoch-30-0.00.hdf5')

        else:

            self.autoencoder = load_model(self.load_model_path)

        # Loss function
        def custom_loss(y_true, y_pred):

            # Define losses
            if (self.loss[0] == 'mse'):
                rloss1 = mean_squared_error(K.flatten(y_true), K.flatten(y_pred))
            else:
                rloss1 = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
            if (self.loss[1] == 'mse'):
                rloss2 = mean_squared_error(K.flatten(l1), K.flatten(l2))
            else:
                rloss2 = binary_crossentropy(K.flatten(l1), K.flatten(l2))

            # Combine losses and add to model
            custom_loss = K.mean(rloss1 + rloss2)
            return custom_loss

        # Compile model with custom loss
        self.autoencoder.compile(optimizer=self.optimizer, loss=custom_loss)


    def build_vae(self):
        assert(self.timesteps % self.pool_size == 0)

        # Encoder
        x1 = Input(shape=(self.timesteps, self.input_dim), name='encoder_input')
        encoded = Conv1D(self.n_filters, self.kernel_size, strides=self.strides, padding='same', activation='linear')(x1)
        encoded = LeakyReLU()(encoded)
        encoded = MaxPool1D(self.pool_size)(encoded)

        for i in range(len(self.n_units)):
            encoded = Bidirectional(LSTM(self.n_units[i], return_sequences=True), merge_mode='sum')(encoded)
            encoded = LeakyReLU()(encoded)

        encoded_shape = K.int_shape(encoded)
        encoded = Flatten()(encoded)

        z_mean = Dense(self.latent_dim, name='z_mean')(encoded)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(encoded)
        z = Lambda(self.sampling, name='z')([z_mean, z_log_var])

        # Create encoder model
        self.encoder = Model(inputs=x1, outputs=[z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        decoded = Dense(self.tuple_product(encoded_shape), activation='linear')(latent_input)
        decoded = LeakyReLU()(decoded)
        decoded = Reshape(encoded_shape[1:])(decoded)

        for i in range(len(self.n_units)):
            decoded = Bidirectional(LSTM(self.n_units[len(self.n_units)-1-i], return_sequences=True), merge_mode='sum')(decoded)
            decoded = LeakyReLU()(decoded)

        decoded = Reshape((-1, 1, self.n_units[0]), name='reshape')(decoded)
        decoded = UpSampling2D((self.pool_size, 1), name='upsampling')(decoded)
        decoded = Conv2DTranspose(self.input_dim, (self.kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
        decoder_output = Reshape((-1, self.input_dim), name='output_seq')(decoded)

        # Create decoder model
        self.decoder = Model(inputs=latent_input, outputs=decoder_output, name='decoder')

        # Connect components
        l1 = self.encoder(x1)
        x2 = self.decoder(l1[2])
        l2 = self.encoder(x2)

        # Create VAE model
        self.autoencoder = Model(inputs=x1, outputs=x2, name='autoencoder')

        # Loss function
        def custom_loss(y_true, y_pred):

            # Define losses
            if (self.loss[0] == 'mse'):
                rloss1 = mean_squared_error(K.flatten(y_true), K.flatten(y_pred))
            else:
                rloss1 = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
            if (self.loss[1] == 'mse'):
                rloss2 = mean_squared_error(K.flatten(l1[0]), K.flatten(l2[0])) + mean_squared_error(K.flatten(l1[1]), K.flatten(l2[1]))
            else:
                rloss2 = binary_crossentropy(K.flatten(l1[0]), K.flatten(l2[0])) + binary_crossentropy(K.flatten(l1[1]), K.flatten(l2[1]))
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

            # Combine losses and add to model
            custom_loss = K.mean(rloss1 + rloss2 + kl_loss)
            return custom_loss

        # Compile model with custom loss
        self.autoencoder.compile(optimizer=self.optimizer, loss=custom_loss)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim)) # by default, random_normal has mean=0 and std=1.0
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def tuple_product(self, tuple):
        prod = 1
        for i in range(len(tuple)):
            if tuple[i] is not None:
                prod = tuple[i] * prod
        return prod
