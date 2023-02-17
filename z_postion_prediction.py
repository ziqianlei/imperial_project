# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:52:15 2019

@author: ziqian

This is used for z-position prediction
"""
import skimage as skim
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

for ii in range(1,40):
    path = "C:/Users/ziqian/Desktop/asti/crop51_1/"+str(ii)+".tif" #saved all cropped single-bead images 32*32*51
    I = skim.io.imread(path)
    Ishape = np.shape(I)
    I_x = []
    I_y = []
    Ast_mag = []
    for ii in range(Ishape[0]):
        #I_x.append((I[ii]-beadmin)/(beadmax-beadmin)) normalization
        I_y.append(ii*0.04-1)
        I_x.append(I[ii]/65535)
    
    training_data = (I_x, I_y)
    I_x_nor = np.array(I_x)
    
    Ast_mag = I_x_nor[0, :, :]
    Ast_mag = Ast_mag.reshape(1, 32, 32)
    
    save_path = r"E:\A_files\optics_and_photonics\summer_project\summerproject\figures\z_pos\0705\zpos_model_trained" 
    modelLoad = tf.keras.models.load_model(save_path)
    
    # predict
    
    #for jj in range(Ishape[0]):
       # Ast_mag = I_x_nor[jj, :, :]
    #Ast_mag = Ast_mag.reshape(1, 32, 32)
    #predictions_ast = modelLoad.predict(Ast_mag)
       # print(jj, "th predict:", predictions_ast)
    
    predictions = modelLoad.predict(I_x_nor)
    
    #plot predictions
    plt.plot(I_y,predictions,'o')
    plt.plot(I_y, I_y)
    plt.ylabel("prediction /um")
    plt.xlabel("experiment /um")
    title = "Z_position"
    plt.title(title)
    plt.savefig("C:/Users/ziqian/Desktop/Aberration_error/" + title+str(ii)+".png")
    plt.clf