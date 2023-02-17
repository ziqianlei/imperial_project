# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:24:57 2019

@author: ziqian

Building and training z-postion model 
"""
import scipy.io as sio
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, PReLU
from tensorflow.keras.callbacks import TensorBoard


path = "E:/A_files/optics_and_photonics/summer_project/summerproject/figures/z_pos/0911/"
train_path = path + "train/" 
direc_train = [name for name in os.listdir(train_path)]
Num_train = len(direc_train)
Num = 51

#training_data = []
x_train = []
y_train = []
for order in range(Num_train):
    filename = train_path + direc_train[order]
    dic = sio.loadmat(filename)
    image = dic['psf_noi']
    for ii in range(Num):
        x_train.append(image[:, :, ii])
        y_train.append(ii*0.04-1) #from -1 to 1
    
    #print (np.shape(x_train))=(Num_train*Num, 32, 32)
    #print (np.shape(y_train))=(Num_train*Num,)
x_train = np.array(x_train)/255.0
training_data = (x_train, y_train)

#%%GenTrainModel

model = tf.keras.models.Sequential()
model.add(Flatten(input_shape=x_train.shape[1:])) 
#model.add(Dense(256, activation=LeakyReLU))
model.add(Dense(256))
model.add(PReLU())
#model.add(Dense(256,activation=tf.nn.tanh))
model.add(Dense(128))
model.add(PReLU())
model.add(Dense(64))
model.add(PReLU())
model.add(Dense(32))
model.add(PReLU())
model.add(Dense(1))

SGD=tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error',
              optimizer=SGD,
              metrics=['mse'])

model.fit(x_train, y_train, batch_size=16, validation_split=0.25, epochs=5) 
save_path = path + "zpos_model_trained" 
model.save(save_path)

#%%GenTestData
test_path = path + "test/"
direc_test = [name for name in os.listdir(test_path)]
Num_test = len(direc_test)
Num = 51
x_test = []
y_test = []
for order in range(Num_test):
    filename = test_path + direc_test[order]
    dic = sio.loadmat(filename)
    image = dic['psf_noi']
    for ii in range(Num):
        x_test.append(image[:, :, ii])
        y_test.append(ii*0.04-1) #from 0 to Num-1,
   
x_test1 = np.array(x_test)/255.0
test_data = (x_test1, y_test)

#%% test
modelLoad = tf.keras.models.load_model(save_path)
# predict
predictions = modelLoad.predict(x_test1)

# plot
plt.plot(y_test,predictions, 'o')
plt.ylabel("prediction /um")
plt.xlabel("experiment /um")
title = "Z position testing data"
