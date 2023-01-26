import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from scipy.io import arff

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import sys
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, './')
import _MAIN_Regression as Regression

###################################
############ PLOTS ################
###################################
    
def plotPrediction(ys, i, delta=None):
    
    all_y = ys[i]
    
    test_predicted = ys[i][1]
    test_real = ys[i][0]

    fig, ax = plt.subplots(figsize=(8, 5))       
    
    ax.set_xlabel(r'$y_1$', fontsize=20)
    ax.set_ylabel(r'$y_2$', fontsize=20)
    ax.set_xlim([0, 18])
    ax.set_ylim([0, 350])
                 
    s=200    
    for i in range(len(test_real)):
        xi = [test_real.iloc[i,0], test_predicted.iloc[i,0]]
        yi = [test_real.iloc[i,1], test_predicted.iloc[i,1]]
        ax.plot(xi, yi,  c='black', alpha=0.1)
    ax.scatter(test_real.iloc[:,0], test_real.iloc[:,1], 
                c='#FFB139', alpha=0.6, s=s,
                label='true values')
    ax.scatter(test_predicted.iloc[:,0], test_predicted.iloc[:,1], 
                c='#630C3A', alpha=0.6, s=s,
                label='predicted values')

    
    if delta.shape[0] == 2:
        cy1, cy2 = 4, 50
        ellipse = ptch.Ellipse((cy1, cy2), delta[0]/2, delta[1]/2, color='black', fill=False, ls='--')
        ax.plot((cy1), (cy2), '.', color='black')
        ax.add_patch(ellipse)    

    ax.legend(prop={"size":15})  
    
    return fig




###################################
############ DATA #################
###################################
    
import pyreadr

Data = pyreadr.read_r('./Data/yacon.rda') # also works for Rds, rda
Data = Data["yacon"] 
Data = Data.drop(columns=['locality', 'dose', 'entry']) # drop categorical
targets = ['brix', 'height']
### useless X features only ###
Data = Data[targets]
Data['X'] = np.random.standard_normal(len(Data))
name = 'yacon_BH'

###########################################
############ TRAINING PART ################
###########################################

path_res = './Results/Yacon/'

delta = 1.0


logi, ys, delta_inv = Regression.compareModels(Data, targets, delta=delta, clusters=5)    
logi.to_csv(path_res + name + '_logi.csv')

### Save plots
pdf = matplotlib.backends.backend_pdf.PdfPages(path_res + name + '_ys.pdf')
for i in range(len(ys)):
    pdf.savefig(plotPrediction(ys,i, delta_inv))
pdf.close()


