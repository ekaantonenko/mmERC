

This is a code implementation of the algorithm from the paper https://doi.org/10.1007/978-3-031-01333-1_1,
you any questions can contact me via <em>ekaterina \<dot\> antonenko \<at\> polytechnique \<dot\> edu</em>

# Multi-modal Ensembles of Regressor Chains

There are two models implemented:
- Ensembles of Regressor Chains (ERC)
- Multi-modal Ensembles of Regressor Chains (mmERC)

## Structure

- MultiModalRegression.py: implementation of ERC and mmERC
- _MAIN_Regression.py: comparison and cross validation of different models
- __MAIN__datasetname.py: executable file for collecting metrics and plots 
- ScoresMulti.py: metrics UCF, aRRMSE, correntropy
- ModelsChains.py: Base estimators for chains + standard models to compare with  
  

## Usage

```python

import MultiModalRegression as MMRC

### mmERC
# Hyperparameters:
# n = number of chains in ensemble // default: 10
# sample_size = part of data to choose on 2nd step of training phase // default: 0.5
# clusters = number of clusters for KMeans algorythm // default: 2
mmERC = MMRC.EnsembleRegressorChains_MultiModal(mdl, 
                                                n=n,
                                                sample_size=sample_size,
                                                clusters=clusters)
mmERC.fit(X_train, y_train)
y_pred = mmERC.predict(X_test)

### ERC
# Hyperparameters:
# n = number of chains in ensemble // default: 10
ERC = MMRC.EnsembleRegressorChains(mdl, 
                                   n=n)
ERC.fit(X_train, y_train)
y_pred = ERC.predict(X_test)                               
                                               
```
