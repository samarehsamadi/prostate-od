from TAE import temporal_autoencoder, temporal_autoencoder_v3
from utils import dataloader, visualizer
from DeepTemporalClustering import DTC

from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error

from keras.callbacks import History
from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import os
import matplotlib.pyplot as plt
import numpy as np

class AE_only:

    def __init__(self,
                 # Data loading options
                 dataset='extended',
                 norm_method='min_max_per_core',
                 val_fold=1,
                 crop=None,
                 inv_thresh=0.4,
                 custom_data='benign',
                 # Network setup
                 n_filters=32,
                 kernel_size=8,
                 strides=1,
                 pool_size=4,
                 n_units=[16, 8, 4],
                 latent_dim=10,
                 # Compiler options
                 loss=mean_squared_error,
                 optimizer='adam',
                 # Training options
                 epochs=100,
                 batch_size=256):

        # Arguments
        self.dataset = dataset
        self.norm_method = norm_method
        self.val_fold = val_fold
        self.crop = crop
        self.inv_thresh = inv_thresh
        self.custom_data = custom_data
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
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.dl = None

    def visualize(self, num_examples=5, max_samples=10000, hist_range=None):

        vs = visualizer(model=self.model,
                        history=self.history,
                        dataset=self.dataset,
                        norm_method=self.norm_method,
                        val_fold=self.val_fold,
                        crop=self.crop,
                        inv_thresh=self.inv_thresh)
        vs.training_curve()
        vs.benign_cancer_examples(num_examples)
        vs.error_distribution(max_samples, hist_range)

    def fit(self):

        # Prepare data
        self.dl = dataloader(dataset=self.dataset,
                             norm_method=self.norm_method,
                             val_fold=self.val_fold,
                             crop=self.crop,
                             inv_thresh=self.inv_thresh,
                             custom_data=self.custom_data,
                             verbose=2)

        # Init network
        self.autoencoder, self.encoder, self.decoder = temporal_autoencoder_v3(input_dim=1,
                                                                               timesteps=self.dl.data_train.shape[1],
                                                                               n_filters=self.n_filters,
                                                                               kernel_size=self.kernel_size,
                                                                               strides=self.strides,
                                                                               pool_size=self.pool_size,
                                                                               n_units=self.n_units,
                                                                               latent_dim=self.latent_dim)
        # DTC model (autoencoder only)
        self.model = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.output)

        # Compile model
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        # Prepare callbacks
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.000)
        filepath = "model-ae-epoch-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=False, period=10)

        # Fit model
        self.history = self.model.fit(x=self.dl.data_train,
                                      y=self.dl.data_train,
                                      validation_data=[self.dl.data_val, self.dl.data_val],
                                      nb_epoch=self.epochs,
                                      batch_size=self.batch_size,
                                      verbose=2,
                                      callbacks=[early_stop, checkpoint]).history

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def predict(self, x):
        return self.model.predict(x)

class full_network:

    def __init__(self,
                 # Data loading options
                 dataset='extended',
                 norm_method='min_max_per_core',
                 val_fold=1,
                 crop=None,
                 inv_thresh=0.4,
                 custom_data='benign+hi_inv',
                 # AE network setup
                 n_filters=100,
                 kernel_size=10,
                 strides=1,
                 pool_size=5,
                 n_units=[100,2],
                 # Clustering network setup
                 n_clusters=2,
                 cluster_init='kmeans',
                 dist_metric='eucl',
                 gamma=1.0,
                 # Heatmap network setup
                 heatmap=False,
                 finetune_heatmap_at_epoch=8,
                 initial_heatmap_loss_weight=0.1,
                 final_heatmap_loss_weight=0.9,
                 alpha=1.0,
                 # Compiler options
                 optimizer='adam',
                 # Training options
                 pretrain_epochs=10,
                 epochs=100,
                 eval_epochs=1,
                 save_epochs=10,
                 batch_size=512,
                 save_dir='/workspace/workspace/dtc-ae/results'):

        # Check if save_dir is valid
        if (not os.path.exists(save_dir)):
            raise ValueError('Save directory ~save_dir~ does not exist')

        # Arguments
        self.dataset = dataset
        self.norm_method = norm_method
        self.val_fold = val_fold
        self.crop = crop
        self.inv_thresh = inv_thresh
        self.custom_data = custom_data

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units

        self.n_clusters = n_clusters
        self.cluster_init = cluster_init
        self.dist_metric = dist_metric
        self.gamma = gamma

        self.heatmap = heatmap
        self.finetune_heatmap_at_epoch = finetune_heatmap_at_epoch
        self.initial_heatmap_loss_weight = initial_heatmap_loss_weight
        self.final_heatmap_loss_weight = final_heatmap_loss_weight
        self.alpha = alpha

        self.optimizer = optimizer

        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.eval_epochs = eval_epochs
        self.save_epochs = save_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir

        # Initialize
        self.model = None

    def visualize(self, num_examples=5, max_samples=10000, hist_range=None):

        vs = visualizer(model=self.model,
                        history=None,
                        dataset=self.dataset,
                        norm_method=self.norm_method,
                        val_fold=self.val_fold,
                        crop=self.crop,
                        inv_thresh=self.inv_thresh)

        vs.benign_cancer_examples(num_examples)

    def fit(self):

        # Load data
        self.dl = dataloader(dataset=self.dataset,
                             norm_method=self.norm_method,
                             val_fold=self.val_fold,
                             crop=self.crop,
                             inv_thresh=self.inv_thresh,
                             custom_data=self.custom_data,
                             verbose=2)

        # Init network
        self.dtc = DTC(n_clusters=self.n_clusters,
                       input_dim=1,
                       timesteps=self.dl.data_train.shape[1],
                       n_filters=self.n_filters,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       pool_size=self.pool_size,
                       n_units=self.n_units,
                       alpha=self.alpha,
                       dist_metric=self.dist_metric,
                       cluster_init=self.cluster_init,
                       heatmap=self.heatmap)

        # Initialize and compile model
        self.dtc.initialize()
        self.dtc.model.summary()
        self.dtc.compile(gamma=self.gamma,
                         optimizer=self.optimizer,
                         initial_heatmap_loss_weight=self.initial_heatmap_loss_weight,
                         final_heatmap_loss_weight=self.final_heatmap_loss_weight)

        # Pretrain
        if (self.pretrain_epochs > 0):
            self.dtc.pretrain(X=self.dl.data_train,
                              optimizer=self.optimizer,
                              epochs=self.pretrain_epochs,
                              batch_size=self.batch_size,
                              save_dir=self.save_dir)

        # Initialize clusters
        self.dtc.init_cluster_weights(self.dl.data_train)

        # Fit model
        self.dtc.fit(X_train=self.dl.data_train,
                     y_train=self.dl.label_train,
                     X_val=self.dl.data_val,
                     y_val=self.dl.label_val,
                     epochs=self.epochs,
                     eval_epochs=self.eval_epochs,
                     save_epochs=self.save_epochs,
                     batch_size=self.batch_size,
                     tol=0.001,
                     patience=5,
                     finetune_heatmap_at_epoch=self.finetune_heatmap_at_epoch,
                     save_dir=self.save_dir)

        self.model = self.dtc.model

