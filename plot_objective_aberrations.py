# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:26:00 2019

@author: ziqian
"""
import skimage as skim
import pandas as pd
from pandas import ExcelFile
import os
import matplotlib.pyplot as plt
import numpy as np

mode='lowCost'
plow = "C:/Users/ziqian/Desktop/asti/"+mode+"/"
ptable = [plow + name for name in os.listdir(plow) if name.endswith(".xlsx")]

xpos, ypos, sigma, txm, tym, defm, astam, astxm, cmxm, cmym, sphm, txp, typ, defp, astap, astxp, cmxp, cmyp, sphp, txam, tyam, deam, astaam, astxyam, cmxam, cmyam, spham = [pd.Series()]*27
parameters={'x_centre':xpos, 'y_centre':ypos, 'sigma': sigma, 'tilt_x_mag':txm, 'tilt_y_mag':tym, 'defocus_mag':defm, 'ast_ax_mag':astam, 'ast_xy_mag':astxm, 'coma_x_mag':cmxm, 'coma_y_mag':cmym, 'sphe_mag': sphm, 'tilt_x_phase':txp, 'tilt_y_phase':typ, 'defocus_phase':defp, 'ast_ax_phase':astap, 'ast_xy_phase':astxp, 'coma_x_phase':cmxp, 'coma_y_phase':cmyp, 'sphe_phase': sphp  }

# initialization Read stastics from tables
for pa in ptable:
    df = pd.read_excel(pa)
    for name in parameters:
        parameters[name] = parameters[name].append(df[name], ignore_index=True)

# image background  
path = "E:/A_files/optics_and_photonics/summer_project/summerproject/20190812_Ziqian_objectives/lowCost/635_10/"
underfilespath = [path + name for name in os.listdir(path) if name.endswith(".tif")]

ic = skim.io.imread_collection(underfilespath)
stackarray = skim.io.concatenate_images(ic)
img = stackarray[12,:,:]

#%% plot and save figures

for name in parameters:
    #plt.imshow(img,'gray')
    sc=plt.scatter(parameters['x_centre'], parameters['y_centre'], c=parameters[name], s=parameters['sigma'], cmap='RdBu',vmin=-1, vmax=1)
    plt.title(mode+"_"+name)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.colorbar(sc, orientation='vertical')
    plt.savefig("C:/Users/ziqian/Desktop/asti/NA1.3/"+mode+"_RdBu_" + name+".png", dpi=100)
    plt.clf()