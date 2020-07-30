import hdf5storage
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, Dense, Flatten, Reshape
from keras.losses import binary_crossentropy
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

class dataloader:

    def __init__(self, val_fold=1):

        self.parse_data(val_fold)

    def parse_data(self, fold):

        # Extract data
        inputdata = hdf5storage.read(filename='hist_mixed_full.h5')[0][0]
        self.bin_edges = inputdata['data'][0][0]
        data = np.array([x[0] for x in inputdata['data']]) / 500000
        label = np.array(inputdata['label'])
        patient_id = np.array(inputdata['PatientId'])
        inv = np.array(inputdata['inv'])

        # Allocate train vs test
        test_id = [5, 6, 18, 45, 37, 30, 12, 14, 24, 46, 52, 58, 79, 71]
        test_idx = [True if pid in test_id else False for pid in patient_id]
        train_idx = [not tr for tr in test_idx]

        # Allocate train vs val
        if (fold == 1):
            fold1_id = [3, 4, 8, 21, 25, 29, 64, 80]
            train_idx = [False if pid in fold1_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold1_id else False for pid in patient_id]
        elif (fold == 2):
            fold2_id = [7, 11, 26, 40, 47, 57, 72, 87]
            train_idx = [False if pid in fold2_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold2_id else False for pid in patient_id]
        elif (fold == 3):
            fold3_id = [13, 19, 27, 59, 60, 65, 81, 85]
            train_idx = [False if pid in fold3_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold3_id else False for pid in patient_id]
        elif (fold == 4):
            fold4_id = [22, 23, 38, 39, 48, 68, 74, 82]
            train_idx = [False if pid in fold4_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold4_id else False for pid in patient_id]
        elif (fold == 5):
            fold5_id = [2, 10, 28, 42, 66, 70, 76, 89, 90]
            train_idx = [False if pid in fold5_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold5_id else False for pid in patient_id]
        else:
            raise ValueError('Unexpected value for parameter ~val_fold~ (expected 1,2,...,5)')

        # Separate train vs test set
        self.data_train = data[train_idx]
        self.label_train = label[train_idx]
        self.inv_train = inv[train_idx]
        self.pid_train = patient_id[train_idx]
        self.data_val = data[val_idx]
        self.label_val = label[val_idx]
        self.inv_val = inv[val_idx]
        self.pid_val = patient_id[val_idx]
        self.data_test = data[test_idx]
        self.label_test = label[test_idx]
        self.inv_test = inv[test_idx]
        self.pid_test = patient_id[test_idx]

        # Reshape data
        # self.data_train = np.reshape(self.data_train, (self.data_train.shape[0], self.data_train.shape[1], 1))
        # self.data_val = np.reshape(self.data_val, (self.data_val.shape[0], self.data_val.shape[1], 1))
        # self.data_test = np.reshape(self.data_test, (self.data_test.shape[0], self.data_test.shape[1], 1))
        print(self.data_train.shape)
        print(self.data_val.shape)
        print(self.data_test.shape)

class histclass:

    def __init__(self, val_fold=2, n_filters=[20, 5], kernel_size=[10, 5], strides=[1, 1], pool_size=[5, 2], optimizer='adam', epochs=100, batch_size=16):

        self.val_fold = val_fold
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self):

        # Boilerplate
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # Init network
        self.dl = dataloader(val_fold=self.val_fold)
        self.build_model()
        self.classifier.summary()

        # Prepare callbacks
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.000)
        filepath = "model-cls-epoch-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=False, period=10)

        # Fit
        self.history = self.classifier.fit(x=self.dl.data_train, y=self.dl.label_train, validation_data=[self.dl.data_val, self.dl.label_val], nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1).history

    def build_model(self):

        num_layers = len(self.n_filters)
        assert(len(self.kernel_size) == num_layers and len(self.strides) == num_layers)

        # Input layer
        x1 = Input(shape=(50,), name='modelinput')

        # Conv layers
        x = Reshape((50, 1))(x1)
        for i in range(len(self.n_filters)):
            x = Conv1D(self.n_filters[i], self.kernel_size[i], strides=self.strides[i], padding='same', activation='linear', kernel_regularizer=l1_l2(0.05, 0.10), bias_regularizer=l1_l2(0.10, 0.05), activity_regularizer=l1_l2(0.05, 0.10))(x)
            x = LeakyReLU()(x)
            x = MaxPool1D(self.pool_size[i])(x)

        # Fully connected layer
        features = Flatten()(x)
        features = Dense(250, activation='relu', kernel_regularizer=l1_l2(0.05, 0.10), bias_regularizer=l1(0.5), activity_regularizer=l1_l2(0.05, 0.10))(features)
        output = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(0.25, 0.50), bias_regularizer=l1(0.8))(features)

        # Define model
        self.classifier = Model(inputs=x1, outputs=output, name='classifier')

        # Specificity and sensitivity metric functions
        def sensitivity(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            return true_positives / (possible_positives + K.epsilon())

        def specificity(y_true, y_pred):
            true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
            possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
            return true_negatives / (possible_negatives + K.epsilon())

        # Compile model
        self.classifier.compile(optimizer=self.optimizer, loss=binary_crossentropy, metrics=['accuracy', sensitivity, specificity])

