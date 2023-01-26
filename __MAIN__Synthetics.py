import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import _MAIN_Regression as Regression


###################################
############ DATA #################
###################################
    
Datasets = []
#Datasets += ['synth_0' + str(i) + '_A' for i in range(1,9)]
Datasets += ['synth_0' + str(i) + '_B' for i in range(1,9)]
Datasets += ['synth_0' + str(i) + '_C' for i in range(1,9)]
Datasets += ['synth_0' + str(i) + '_D' for i in range(1,9)]
Datasets += ['synth_0' + str(i) + '_E' for i in range(1,9)]

###########################################
############ TRAINING PART ################
###########################################

path_data = './Data/Synthetic/'
path_res = './Results/Synthetic/'

delta = 1.0
for name in Datasets:
    
    print(name)
    Data = pd.read_csv(path_data + name + '.csv')
    targets = ['Y1', 'Y2']
    logi, ys, delta_inv = Regression.compareModels(Data, targets, delta=delta)    
    logi.to_csv(path_res + name + '.csv')
    

    

    
