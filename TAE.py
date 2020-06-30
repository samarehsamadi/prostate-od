"""
Implementation of the Deep Temporal Clustering model
Temporal Autoencoder (TAE)

@author Florent Forest (FlorentF9)
"""

from keras.models import Model
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D,LSTM, CuDNNLSTM, Bidirectional, TimeDistributed, Dense, Reshape
from keras.layers import Flatten, Lambda
from keras.layers import UpSampling2D, Conv2DTranspose
from keras import backend as K


def temporal_autoencoder(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):
    """
    Temporal Autoencoder (TAE) model with Convolutional and BiLSTM layers.

    # Arguments
        input_dim: input dimension
        timesteps: number of timesteps (can be None for variable length sequences)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences

    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    assert(timesteps % pool_size == 0)

    # Input
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)
    encoded = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='sum')(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='sum')(encoded)
    encoded = LeakyReLU(name='latent')(encoded)

    # Decoder
    decoded = Reshape((-1, 1, n_units[1]), name='reshape')(encoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)  #decoded = UpSampling1D(pool_size, name='upsampling')(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)  #output = Conv1D(1, kernel_size, strides=strides, padding='same', activation='linear', name='output_seq')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(timesteps // pool_size, n_units[1]), name='decoder_input')

    # Internal layers in decoder
    decoded = autoencoder.get_layer('reshape')(encoded_input)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder


def temporal_autoencoder_v2(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):
    """
    Temporal Autoencoder (TAE) model with Convolutional and BiLSTM layers.

    # Arguments
        input_dim: input dimension
        timesteps: number of timesteps (can be None for variable length sequences)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences

    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    assert (timesteps % pool_size == 0)

    # Input
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)
    encoded = Bidirectional(CuDNNLSTM(n_units[0], return_sequences=True), merge_mode='concat')(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Bidirectional(CuDNNLSTM(n_units[1], return_sequences=True), merge_mode='concat')(encoded)
    encoded = LeakyReLU(name='latent')(encoded)

    # Decoder
    decoded = TimeDistributed(Dense(units=n_filters), name='dense')(encoded)  # sequence labeling
    decoded = LeakyReLU(name='act')(decoded)
    decoded = Reshape((-1, 1, n_filters), name='reshape')(decoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(timesteps // pool_size, 2 * n_units[1]), name='decoder_input')
    # Internal layers in decoder
    decoded = autoencoder.get_layer('dense')(encoded_input)
    decoded = autoencoder.get_layer('act')(decoded)
    decoded = autoencoder.get_layer('reshape')(decoded)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder

def temporal_autoencoder_v3(input_dim, timesteps, n_filters=32, kernel_size=16, strides=1, pool_size=8, n_units=[16, 8, 4], latent_dim=10):
    """
    Temporal Autoencoder (TAE) model with Convolutional and BiLSTM layers.

    # Arguments
        input_dim: input dimension
        timesteps: number of timesteps (can be None for variable length sequences)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences

    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    assert(timesteps % pool_size == 0)

    # Input
    input = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(input)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)

    for i in range(3):
        encoded = Bidirectional(LSTM(n_units[i], return_sequences=True), merge_mode='sum')(encoded)
        encoded = LeakyReLU()(encoded)

    encoded_shape = K.int_shape(encoded)
    print(encoded_shape)
    print(tuple_product(encoded_shape))
    encoded = Flatten()(encoded)
    z_mean = Dense(latent_dim, name='z_mean')(encoded)
    z_log_var = Dense(latent_dim, name='z_log_var')(encoded)
    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    # Encoder model
    encoder = Model(inputs=input, outputs=[z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_input = Input(shape=(latent_dim,), name='z_sampling')
    decoded = Dense(tuple_product(encoded_shape), activation='linear')(latent_input)
    decoded = LeakyReLU()(decoded)
    decoded = Reshape(encoded_shape[1:])(decoded)

    for i in range(3):
        decoded = Bidirectional(LSTM(n_units[2-i], return_sequences=True), merge_mode='sum')(decoded)
        decoded = LeakyReLU()(decoded)

    decoded = Reshape((-1, 1, n_units[0]), name='reshape')(decoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    decoder_output = Reshape((-1, input_dim), name='output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=latent_input, outputs=decoder_output, name='decoder')

    # VAE model
    output = decoder(encoder(input)[2])
    autoencoder = Model(inputs=input, outputs=output, name='autoencoder')

    return autoencoder, encoder, decoder

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def tuple_product(tuple):
    prod = 1
    for i in range(len(tuple)):
        if tuple[i] is not None:
            prod = tuple[i] * prod
    return prod

