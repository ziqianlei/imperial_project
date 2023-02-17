# -*- coding: utf-8 -*-
"""
Cropping, normalization and prediction

Created on Wed Aug 14 17:02:44 2019

@author: ziqian
"""

import skimage as skim
import skimage.feature
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

# path of objective images folder
p = "E:/A_files/optics_and_photonics/summer_project/summerproject/20190830_Ziqian_objectives/standard/"
stackorder = [n for n in os.listdir(p)]

for ii in range(0,10):
    test=stackorder[ii]
    path = p + test+"/"
    underfilespath = [path + name for name in os.listdir(path) if name.endswith(".tif")] #get images folder names of every acquisition
        
    ic = skim.io.imread_collection(underfilespath)
    stackarray = skim.io.concatenate_images(ic)
    img = stackarray[13,:,:] #estimated focal plane
    
    # adjust the contrast for beads localization
    max_input = 5000 #LowCost: 3000; standard and TIRF: 5000
    min_input = np.amin(img)
    max_output = 1
    min_output = 0
    im = (np.array(img)-min_input)*(((max_output-min_output)/(max_input-min_input))+min_output)
    
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    minsigma = 0.5
    maxsigma = 3
    thres = 0.5 
    
    blob_log= skim.feature.blob_log(im, min_sigma=minsigma, max_sigma=maxsigma, threshold=thres)
    
    for ii in range(blob_log.shape[0]):
        y, x, r = blob_log[ii]
        if 0.5<=r<=2:
            rect = patches.Rectangle((x-16, y-16), 32, 32, edgecolor='r', fill=None)
            ax.add_patch(rect)
    plt.show()
    
    [centre_y, centre_x, size] = np.hsplit(blob_log,3)
    X=[] # save x_position after filtering
    Y=[] # save y_position after filtering
    Size=[]
    def giverange(order, totalbeads):
        if order<40:
            return range(-order, 40)
        if order>totalbeads-40:
            return range(-40, (totalbeads-order))
        if 40<=order<=(totalbeads-40):
            return range(-40, 40)
    
    # filter images with multiple beads
    for ii in range(blob_log.shape[0]):
        overlap_flag = 0
        for xx in giverange(ii, blob_log.shape[0]): 
            if abs(centre_x[ii]-centre_x[xx+ii])<=32 and abs(centre_y[ii]-centre_y[xx+ii])<=32 and xx!=0:
                overlap_flag += 1 
        if overlap_flag == 0 and 0.5<=size[ii]<=2 and 16<centre_x[ii]<1184 and 16<centre_y[ii]<1184:
           X.append(int(np.asscalar(centre_x[ii])))
           Y.append(int(np.asscalar(centre_y[ii])))
           Size.append(np.asscalar(size[ii]))
    
    
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    for jj in range(len(X)):
        rect = patches.Rectangle((X[jj]-16, Y[jj]-16), 32, 32, edgecolor='r', fill=None)
        ax.add_patch(rect)
    plt.show()
    
    # save all cropped images
    
    crop_bead = []
    
    for ii in range(len(X)):
        crop_bead.append(stackarray[:,Y[ii]-16:Y[ii]+16,X[ii]-16:X[ii]+16])
    
    # find focal plane
    focal_plane = 13 # 
    
    # cropping and normalization

    cropbead_new=[]
    cropbead_nor=[]
    beadmax=0
    beadmin=65535
    strong=[]
    weak = []
    for ii in range(len(X)):
        singlebead = stackarray[focal_plane-6:focal_plane+5,Y[ii]-16:Y[ii]+16,X[ii]-16:X[ii]+16]

        #find max and min value and do normalization
        slidemax = np.amax(singlebead)
        slidemin = np.amin(singlebead)
        beadmax = max(beadmax, slidemax)
        beadmin = min(beadmin, slidemin)
        cropbead_new.append(singlebead) #16bits beads images
        cropbead_nor.append(np.array(singlebead)/65535) #nomalized beads images
        
    
    # make predictions
    array = np.asarray(cropbead_nor)
    arraynew = np.transpose(cropbead_nor, (0,2,3,1))
    
    model_path = r"C:\Users\ziqian\OneDrive - Imperial College London\summer project\neutral_network\NA1_4"
    modelload = tf.keras.models.load_model(model_path)
    predictions = modelload.predict(arraynew)
    name = ["tilt_x_phase", "tilt_y_phase","defocus_phase","ast_ax_phase", "ast_xy_phase", "coma_x_phase", "coma_y_phase", "sphe_phase", "tilt_x_mag", "tilt_y_mag", "defocus_mag", "ast_ax_mag", "ast_xy_mag", "coma_x_mag", "coma_y_mag", "sphe_mag"]
    

    # save positions and predictions into excel files
    import xlsxwriter as xlwr
    
    workbook = xlwr.Workbook("C:/Users/ziqian/Desktop/asti/" +test+ ".xlsx")
    worksheet = workbook.add_worksheet(name='Ast'+test)
    
    
    
    for order in range(len(name)):
        worksheet.write(0, order, name[order])
        for ii in range(len(predictions[order])):
            worksheet.write(1+ii, order, predictions[order][ii])
            
    position = ["x_centre", "y_centre", "sigma"]
    for pos in range(3):
        worksheet.write(0, 17+pos, position[pos])
       
    for xx in range(len(X)):
        worksheet.write(1+xx, 17, X[xx])
    
    for yy in range(len(Y)):
        worksheet.write(1+yy, 18, Y[yy])
        
    for ss in range(len(Size)):
        worksheet.write(1+ss, 19, Size[ss])
        
    workbook.close()
    
    #%% plot
    plt.imshow(img,'gray')
    area=Size
    colors=predictions[7].flatten()
    x=np.array(X)
    y=np.array(Y)
    size=np.array(Size)
    sc=plt.scatter(x, y, c=colors,s=size, cmap='jet')
    plt.title("standard_Spherical")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.colorbar(sc, orientation='vertical')
    #plt.savefig("C:/Users/ziqian/Desktop/Aberration_error/" + "sphe_lowCost_1"+".png", dpi=200)
