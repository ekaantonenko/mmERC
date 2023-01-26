### Basics
import pandas as pd
import numpy as np
import copy
import time
from sklearn.base import clone

### Cross Validation, etc.
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
    
### Metrics
from sklearn.metrics import *

#######################
### Local Packages: ###
#######################
import ModelsChains as mdls
import ScoresMulti as scm
import MultiModalRegression as MMRC

   

def compareModels (Data, targets,
                  order = None,
                  delta = 0.5,
                  clusters=5
                  ):   
     
    X = Data.drop(columns = targets)
    y = Data[targets]
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X = X.to_numpy()
    y = y.to_numpy()
    ##################################################                                                                      
    ################# Models: ########################   
    ##################################################
    
    allModels = []
    
    ### Hyperparameters ###
    sample_size = 0.5 # sample size for mmERC models
    n = 10 # Number of random chains in ensemble
    
    ### mmERC ###
    allModels += [MMRC.EnsembleRegressorChains_MultiModal(mdl, 
                                                          n=n,
                                                          sample_size=sample_size,
                                                          clusters=clusters)
                                              for mdl in mdls.innerModelsRegression()
                                              ]
    ### ERC ###
    allModels += [MMRC.EnsembleRegressorChains(mdl, 
                                               n=n)
                                              for mdl in mdls.innerModelsRegression()
                                              ]
    
    
    ### sklearn models ###
    allModels += mdls.modelsRegression()
    
    
    ##################################################                                                                      
    ############### Prediction: ######################   
    ##################################################  
    
    ### Assigning folds ### 
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    idxs_train, idxs_test = [], []
    for train_index, test_index in kf.split(X):
        idxs_train.append(train_index)
        idxs_test.append(test_index)

        
    
    ### logi ###
    logi = []
    ys = []
    
    ### Test models ###
    for mdl in allModels: 
        
        name = mdl.__class__.__name__
        
        if name == 'MultiOutputRegressor':
            name = str(name) + '_' + str(mdl.estimator.__class__.__name__)
        elif name == 'RegressorChain':
            name = str(name) + '_' + str(mdl.base_estimator.__class__.__name__)
                 
        elif name == 'EnsembleRegressorChains':
            name = 'ERC_' + str(mdl.base_estimator.__class__.__name__) 
        elif name == 'EnsembleRegressorChains_MultiModal':
            name = 'mmERC_' + str(mdl.base_estimator.__class__.__name__) + '_s=' + str(mdl.sample_size)
        elif name == 'singleRC':
            name = 'RC_' + str(mdl.base_estimator.__class__.__name__) 
        
            
        print(name)  
        
        cv_logi = []
        cv_mdls = [clone(mdl) for _ in range(n_splits)]
        delta_inv = []
        
        ### testing on folds ###
        print('Folds: ', end='')
        
        
        all_y_predictions = np.empty(y.shape)
        
        for i in range(n_splits):
            
            print (i, end=' ')

            train_index, test_index = idxs_train[i], idxs_test[i]
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index,:], y[test_index,:]
            
            ### Scale data ###
            scalerX = StandardScaler()
            X_train = scalerX.fit_transform(X_train)
            X_test = scalerX.fit_transform(X_test)
            
            scalerY = StandardScaler()
            y_train = scalerY.fit_transform(y_train)
            y_test = scalerY.fit_transform(y_test)       

            ##################
            
            cv_mdl = cv_mdls[i]
            
            fit_time = 0
            score_time = 0
            
            ### fit ###
            start_time = time.time()
            cv_mdl.fit(X_train, y_train)
            fit_time += (time.time() - start_time)
            
            ### predict train ###
            y_prediction_train = cv_mdl.predict(X_train)
            
            
            ### predict test ###
            start_time = time.time()
            y_prediction_test = cv_mdl.predict(X_test)
            score_time += (time.time() - start_time)
   
            y_prediction_test_inv = scalerY.inverse_transform(y_prediction_test)
            

            y_test_inv = scalerY.inverse_transform(y_test)

            
            ### scores ###
            ucf_train = scm.ucf_score(y_train, y_prediction_train, delta=delta)
            ucf_test = scm.ucf_score(y_test, y_prediction_test, delta=delta)
            aRRMSE_train = scm.aRRMSE(y_train, y_prediction_train)
            aRRMSE_test = scm.aRRMSE(y_test, y_prediction_test)
            
            correntropy_train = scm.correntropy(y_train, y_prediction_train, h=1)
            correntropy_test = scm.correntropy(y_test, y_prediction_test, h=1)

            cv_logi.append([fit_time, score_time, 
                            ucf_train, ucf_test, 
                            aRRMSE_train, aRRMSE_test,
                            correntropy_train, correntropy_test
                            ])
        
            delta_var = scalerY.var_
            delta_inv.append(np.multiply([delta,delta], np.sqrt(delta_var)))
            
            
            all_y_predictions[test_index] = y_prediction_test_inv
            
        print()
        
        
        cv_logi_mean = np.mean(cv_logi, axis=0)
        cv_logi_var = np.var(cv_logi, axis=0)
        cv_logi = [name, *cv_logi_mean, *cv_logi_var]
        
        
        ys.append([y, all_y_predictions])
        logi.append(cv_logi)
        
      
    ################################################## 
    ################################################## 
    
    logi = pd.DataFrame(logi)       
    logi.columns = ['Regressor', 
                    
            'fit_time', 'score_time', 
            'ucf_train', 'ucf_test', 
            'aRRMSE_train', 'aRRMSE_test',
            'correntropy_train', 'correntropy_test',
            
            'fit_time_var', 'score_time_var', 
            'ucf_train_var', 'ucf_test_var', 
            'aRRMSE_train_var', 'aRRMSE_test_var',
            'correntropy_train_var', 'correntropy_test_var'
            ]
    
    
    logi = logi.set_index('Regressor')
    delta_inv = np.mean(delta_inv, axis=0)
    
    
    return logi, ys, delta_inv






