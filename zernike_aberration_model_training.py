# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:28:09 2019

@author: ziqian
"""
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, add, Dense, PReLU
import scipy.io as sio
import os
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

#%% load path

train_path ="D:/Ziqian_Lei/summer_project/zernike/" + "train/" +"NA1.4/"
marks = ['psf_noi', 'phaseZ', 'magZ']
paths = []
direcs = []
for mark in marks:
    path = train_path + mark +"/"
    paths.append(path)
    direc = [name for name in os.listdir(path)]
    direcs.append(direc)

#check path
if direcs[0] == direcs[1] == direcs[2]:
    numtrain = len(direcs[0])
else:
    print("The files are not matched!")
    exit
    
#%% generate training data
training_data = []
x_train = []
y_train = []

for mark in marks:
    data = []
    path = train_path + mark +"/"
    direc = [name for name in os.listdir(path)]
    for order in range(len(direc)):
        filename = path + direc[order]
        dic = sio.loadmat(filename)
        data.append(dic[mark])
    training_data.append(data)

(psf_trains, phaseZs, magZs) = training_data        
    
x_train = np.array(psf_trains)/255.0
#np.shape(x_train)=(Num_train*Num, 32, 32,3)

#%% get zernike phase and maginitude
# y_train: p1, p2, ...p9 and m1, m2,..., m9; type:list

phases = [[] for i in range(9)]
[p1, p2, p3, p4, p5, p6, p7, p8, p9] = phases
magnitudes = [[] for i in range(9)]
[m1, m2, m3, m4, m5, m6, m7, m8, m9] = magnitudes

# save m1, m2,..., m9 in list m1, m2,..., m9
for magZ in magZs:
    for order in range(len(magnitudes)):
        magnitudes[order].append(magZ[0, order]) # shape of magZ is (1,25)
for order in range(len(magnitudes)):
    magnitudes[order] = np.asarray(magnitudes[order]) 
[m1, m2, m3, m4, m5, m6, m7, m8, m9] = magnitudes

# save p1, p2,..., p9 in list p1, p2,..., p9
for phaseZ in phaseZs:
    for order in range(len(phases)):
        phases[order].append(phaseZ[0, order]) # shape of phaseZ is (1,25)

for order in range(len(phases)):
    phases[order]=np.array(phases[order])
[p1, p2, p3, p4, p5, p6, p7, p8, p9] = phases    



#%% model architecture

train_pm = [p2, p3, p4, p5, p6, p7, p8, p9, m2, m3, m4, m5, m6, m7, m8, m9]

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = './logs' + current_time

inputs = Input(shape=(32, 32, 11), name = 'img_stack')
# first convolution layer
x = Conv2D(64, (7,7), (1,1), padding="same")(inputs)
x = BatchNormalization()(x)
x = PReLU()(x)

# second convolution layer
x = Conv2D(128, (5,5), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)
shortcut_1 = x

# first residual block
x = Conv2D(32, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(64, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(128, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

# block_1_output =!block_2_output
block_1_output = add([x, shortcut_1])
shortcut_2 = Conv2D(256, (1,1), (4,4), padding="valid")(block_1_output)
shortcut_2 = BatchNormalization()(shortcut_2)
shortcut_2 = PReLU()(shortcut_2)

# second residual block
x = Conv2D(64, (3,3), (1,1), padding="same")(block_1_output)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(128, (3,3), (4,4), padding="valid")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(256, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

block_2_output = add([x, shortcut_2])


# third residual block
x = Conv2D(64, (3,3), (1,1), padding="same")(block_2_output)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(128, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(256, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

block_3_output = add([x, block_2_output])

shortcut_3 = Conv2D(1024, (1,1), (4,4), padding="valid")(block_3_output)
shortcut_3 = BatchNormalization()(shortcut_3)
shortcut_3 = PReLU()(shortcut_3)

# forth residual block
x = Conv2D(256, (3,3), (1,1), padding="same")(block_3_output)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(512, (3,3), (4,4), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

x = Conv2D(1024, (3,3), (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = PReLU()(x)

block_4_output = add([x, shortcut_3])


# third convolustion layer
x = Conv2D(1024, (1,1), (1,1), padding="same")(block_4_output)
x = BatchNormalization()(x)
x = PReLU()(x)



x = Flatten()(x)
#x = Dense(64)(x)
#x = PReLU()(x)


#outputs
#p2out = Dense( 1, name = 'tilt_x_phase')(Dense(64, activation = 'relu')(x))
p2out = Dense( 1, name = 'tilt_x_phase')(x)
p3out = Dense( 1, name = 'tilt_y_phase')(x)
p4out = Dense( 1, name = 'defocus_phase')(x)
p5out = Dense( 1, name = 'ast_axis_phase')(x)
p6out = Dense( 1, name = 'ast_xy_phase')(x)
p7out = Dense( 1, name = 'coma_x_phase')(x)
p8out = Dense( 1, name = 'coma_y_phase')(x)
p9out = Dense( 1, name = 'sphe_phase')(x)

m2out = Dense( 1, name = 'tilt_x_mag')(x)
m3out = Dense( 1, name = 'tilt_y_mag')(x)    
m4out = Dense( 1, name = 'defocus_mag')(x)
m5out = Dense( 1, name = 'ast_axis_mag')(x)
m6out = Dense( 1, name = 'ast_xy_mag')(x)
m7out = Dense( 1, name = 'coma_x_mag')(x)
m8out = Dense( 1, name = 'coma_y_mag')(x)
m9out = Dense( 1, name = 'sphe_mag')(x)

models = Model(inputs, outputs=[p2out, p3out, p4out, p5out, p6out, p7out, p8out, p9out, m2out, m3out, m4out, m5out, m6out, m7out, m8out, m9out], name = 'zernike_caculation')
 
tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=logdir)
models.compile(loss={'tilt_x_phase':'mean_squared_error','tilt_y_phase':'mean_squared_error','defocus_phase':'mean_squared_error', 'ast_axis_phase':'mean_squared_error', 'ast_xy_phase':'mean_squared_error', 'coma_x_phase':'mean_squared_error','coma_y_phase':'mean_squared_error', 'sphe_phase':'mean_squared_error','tilt_x_mag':'mean_squared_error','tilt_y_mag':'mean_squared_error', 'defocus_mag':'mean_squared_error', 'ast_axis_mag':'mean_squared_error', 'ast_xy_mag':'mean_squared_error', 'coma_x_mag':'mean_squared_error', 'coma_y_mag':'mean_squared_error', 'sphe_mag':'mean_squared_error'},
                  #optimizer=tf.keras.optimizers.SGD(lr =0.01),
                  optimizer='adam', 
                  loss_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],#all outputs have the same weight
                  metrics=['mse','mse','mse','mse','mse','mse','mse','mse','mse','mse','mse','mse','mse','mse','mse','mse'])

tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=logdir)

models.fit(x_train, {'tilt_x_phase':p2, 'tilt_y_phase':p3, 'defocus_phase':p4, 'ast_axis_phase':p5, 'ast_xy_phase':p6, 'coma_x_phase':p7, 'coma_y_phase':p8, 'sphe_phase':p9, 'tilt_x_mag':m2,'tilt_y_mag':m3,'defocus_mag':m4, 'ast_axis_mag':m5, 'ast_xy_mag':m6, 'coma_x_mag':m7, 'coma_y_mag':m8, 'sphe_mag':m9}, batch_size=16, epochs=50,validation_split=0.01, callbacks=[tensorboard_cbk]) 


save_path = "D:/Ziqian_Lei/summer_project/zernike/NA1_4_0.75"
models.save(save_path)
print("The residual network model has been saved..............")

