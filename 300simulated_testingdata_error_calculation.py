# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:00:22 2019

@author: ziqian
"""

import tensorflow as tf
import numpy as np
import os
import scipy.io as sio

#%% generate testing data
test_path ="E:/A_files/optics_and_photonics/summer_project/summerproject/figures/zernike/test/"
marks = ['psf_noi', 'phaseZ', 'magZ']
paths = []
direcs = []
for mark in marks:
    path = test_path + mark +"/"
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
testing_data = []
x_test = []
y_test = []

for mark in marks:
    data = []
    path = test_path + mark +"/"
    direc = [name for name in os.listdir(path)]
    for order in range(len(direc)):
        filename = path + direc[order]
        dic = sio.loadmat(filename)
        data.append(dic[mark])
    testing_data.append(data)

(psfs, phaseZs, magZs) = testing_data        
    
x_test = np.array(psfs)/255.0

phases = [[] for i in range(9)]
[p1, p2, p3, p4, p5, p6, p7, p8, p9] = phases
magnitudes = [[] for i in range(9)]
[m1, m2, m3, m4, m5, m6, m7, m8, m9] = magnitudes

# save m1, m2,..., m9 in list m1, m2,..., m9
for magZ in magZs:
    for order in range(len(magnitudes)):
        magnitudes[order].append(magZ[0, order]) # shape of magZ is (1,25)
         
# save p1, p2,..., p9 in list p1, p2,..., p9
for phaseZ in phaseZs:
    for order in range(len(phases)):
        phases[order].append(phaseZ[0, order]) # shape of phaseZ is (1,25)

#%% Load model
modelpath = r"C:\Users\ziqian\OneDrive - Imperial College London\summer project\neutral_network\NA1_46"
modelload = tf.keras.models.load_model(modelpath)
predictions = modelload.predict(x_test)

#%% Print predictions as xlsx table
import xlsxwriter as xlwr
workbook = xlwr.Workbook(r"C:\Users\ziqian\Desktop\Aberration_error\Aberration_error.xlsx")
worksheet = workbook.add_worksheet(name='NA_1.46')

aberrationname = ["tilt_x", "tilt_y", "defocus", "astigmatism_axis", "astigmatism_x=y", "coma_x", "coma_y", "spherical" ]

sort = ["Phase", "Magnitude"]

#write column title
worksheet.write(0, 1, "Phase")
worksheet.write(0, 9, "Magnitude")

row = 1
col = 0
for content in sort:
    for name in aberrationname:
        col +=1
        worksheet.write(row, col, name)

# write row title
rowtitle = ["simulation", "prediction", r"Δ"]

for ii in range(x_test.shape[0]):
    for jj in range(len(rowtitle)):
        worksheet.write(ii*3+jj+2, 0, rowtitle[jj])

worksheet.write(3*x_test.shape[0]+2, 0, "MeanSquareError")
worksheet.write(3*x_test.shape[0]+3, 0, "StandardDeviation")

#write simulation, prediction values and error delta
col=0
for order in range(len(predictions)):
    pred = predictions[order]
    if order < 8:
        abre = phases[order+1]
    else:
        abre = magnitudes[order-7]
    col += 1
    row = 2
    sumdelta2 = 0
    for img in range(len(abre)):
        worksheet.write(row, col, abre[img])
        row += 1
        worksheet.write(row, col, pred[img])
        row += 1
        # calculate delta
        delta = pred[img] - abre[img]
        sumdelta2 = sumdelta2 + np.square(delta)
        worksheet.write(row, col, delta)
        row += 1
    mse = sumdelta2/x_test.shape[0]
    worksheet.write(row, col, mse)
    row +=1
    worksheet.write(row, col, np.sqrt(mse))
            
workbook.close()

#%% Print figures
import matplotlib.pyplot as plt


aberrationname = ["tilt_x", "tilt_y", "defocus", "astigmatism_axis", "astigmatism_x=y", "coma_x", "coma_y", "spherical" ]

sort = ["Phase", "Magnitude"]

ii=0
x=[p2, p3, p4, p5, p6, p7, p8, p9, m2, m3, m4, m5, m6, m7, m8, m9]
for typename in sort:
    for name in aberrationname:
        title = name + "_" + typename
        plt.plot(x[ii], predictions[ii], 'ro', x[ii], x[ii], 'b-')
        plt.title(title)
        plt.ylabel("prediction")
        plt.xlabel("simulation")
        ii+=1
        plt.savefig("C:/Users/ziqian/Desktop/Aberration_error/" + title+".png")
        plt.clf()
    


