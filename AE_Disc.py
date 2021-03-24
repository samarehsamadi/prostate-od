# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:46:04 2021

@author: samadi
"""
import os
import keras
import mat73
import numpy as np
import random
from keras.optimizers import Adam
import tensorflow as tf

DataFile="Z:\\shared\\images\\ProstateVGH-2\\Data\\Dataset\\InProstate\\BK_RF_resmp_1_140_balance__20201007-102244.mat"
inputdata = mat73.loadmat(DataFile)
data_train=inputdata["data_train"]

data_val=inputdata["data_val"]

 
batch_size=1024
timesamples=200
n_epochs=10000


def batch_generator(data):
    while True:
        indx=random.sample(range(len(data)), 2)
        indx0=random.sample(range(len(data[indx[0]])),batch_size)
        indx1=random.sample(range(len(data[indx[1]])),batch_size)
        indx2=random.sample(range(len(data[indx[0]])),batch_size)
        data1=np.reshape(data[indx[0]][indx0],(-1,1,200))
        data2=np.reshape(data[indx[1]][indx1],(-1,1,200))
        data3=np.reshape(data[indx[0]][indx2],(-1,1,200))
        data_batch=(data1,data2,data3)
        label_batch=([1]*batch_size,[0]*batch_size,[1]*batch_size)
        yield (data_batch,label_batch)
        
#https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/
model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(1, timesamples)
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=1))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=timesamples)
  )
)

ae_optimizer=Adam(lr=0.0001)

model.compile(loss='mae', optimizer='adam')

def similarity_loss(x,y):
    return tf.keras.losses.mean_squared_error(x,y)

def train_step(x):
        with tf.GradientTape() as ae_tape:
            hatx=model(x[0], training=True)
            loss = similarity_loss(x[0],hatx)

            hatx=model(x[1], training=True)
            loss += similarity_loss(x[1],hatx)

            hatx=model(x[2], training=True)
            loss += similarity_loss(x[2],hatx)

        gradients_of_ae = ae_tape.gradient(loss, model.trainable_variables)
        ae_optimizer.apply_gradients(zip(gradients_of_ae, model.trainable_variables))

        return loss

train_batch = batch_generator(data_train)
val_batch = batch_generator(data_val)

loss_train=[0]*n_epochs
for i in range(n_epochs):
    Xtb, ytb = next(train_batch)
    loss=train_step(Xtb)
    loss_train[i]=np.mean(loss)

    if ((i + 1) % 10 == 0):
        Xvb, yvb = next(val_batch)
        hatx=model(Xvb[0], training=False)
        loss = similarity_loss(Xvb[0],hatx)

        hatx=model(Xvb[1], training=False)
        loss += similarity_loss(Xvb[1],hatx)

        hatx=model(Xvb[2], training=False)
        loss += similarity_loss(Xvb[2],hatx)
        
        log_str = "iter: {:05d}: \nLoss: {:.5f} \n"\
                                             .format(i, np.mean(loss))
        print(log_str)
        model.save(os.path.join("iter_{:05d}_model.h5".format(i)))

plt.plot(loss_train)