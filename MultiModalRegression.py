### Basics
import numpy as np
import copy
from sklearn.base import clone, BaseEstimator


### Metrics
from sklearn.metrics import *

### Methods
from sklearn import cluster
from scipy import stats  




#################################################################################
############### one simple Regressor Chain in ensemble ##########################
#################################################################################

class singleRC(BaseEstimator):
    
    def __init__(self, base_estimator, *, order=None, 
                 random_state=None):
        
        self.base_estimator = base_estimator
        self.order_ = order
        self.random_state = random_state

    ################
    
    def fit(self, X, Y, **fit_params):
    
        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        
        Y_pred_chain = Y
        
        X_aug = np.hstack((X, Y_pred_chain))
        
        for j_idx, j in enumerate(self.order_):           
            y_train = Y[:, j]
            estimator = self.estimators_[j]
            n = X.shape[1]
            ancs = [i + n for i in self.order_[:j_idx] ]
            X_train = X_aug[:, list(range(n)) + ancs]
            estimator.fit(X_train, y_train)
              
        return self            
    
    
    def predict(self, X):

        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))

        for j_idx, j in enumerate(self.order_):
            previous_predictions = Y_pred_chain[:, self.order_[:j_idx] ]
            X_aug = np.hstack((X, previous_predictions))              
            Y_pred_chain[:, j] = self.estimators_[j].predict(X_aug)
            
        return Y_pred_chain   


#################################################################################
############### one Multi-Modal Regressor Chain in ensemble #####################
#################################################################################        
    
class singleRC_MultiModal(BaseEstimator):
    
    def __init__(self, base_estimator, *, order=None, 
                 random_state=None, sample_size=0.5):
        
        self.base_estimator = base_estimator
        self.order_ = order
        self.random_state = random_state
        self.sample_size = sample_size


    ################
        
    def _phi(self, y, h=0.5):
        return np.exp(-(y**2)/(h**2))
    
    def _correntropy(self, y, h=0.5):
        return h**2 * (1 - self._phi(y,h))

    ################
    
    def fit(self, X, Y, **fit_params):
    
        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        
        Y_pred_chain = Y
        
        X_aug = np.hstack((X, Y_pred_chain))
        

        for j_idx, j in enumerate(self.order_): 

            y_train = Y[:, j]
            estimatorB = self.estimators_[j]
            estimatorA = clone(estimatorB)
            n = X.shape[1]
            ancs = [i + n for i in self.order_[:j_idx] ] 
            
            X_train = X_aug[:, list(range(n)) + ancs]
                                
            estimatorA.fit(X_train, y_train)
            y_predict = estimatorA.predict(X_train)
                
                
            y_corr = self._correntropy(y_train - y_predict, 1.0)
            ind_reduced = np.argsort(y_corr)[:int(self.sample_size*len(y_corr))]
            X_train = X_train[ind_reduced]
            y_train = y_train[ind_reduced]
            estimatorB.fit(X_train, y_train)
            
        return self
      
    
    
    def predict(self, X):

        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))

        for j_idx, j in enumerate(self.order_):
            previous_predictions = Y_pred_chain[:, self.order_[:j_idx] ]
            X_aug = np.hstack((X, previous_predictions))              
            Y_pred_chain[:, j] = self.estimators_[j].predict(X_aug)
            
        return Y_pred_chain       
    
    
    
#################################################################################
############### Ensemble of Regressor Chains ####################################
#################################################################################

class EnsembleRegressorChains (BaseEstimator):
    
    def __init__(self, base_estimator, 
                 n=10,
                 random_state=None):
        
        self.base_estimator = base_estimator
        self.n = n
        self.n_outputs = None
        self.random_state = random_state
        
    

    def fit(self, X, Y, **fit_params):
        
        self.n_outputs = Y.shape[1]
        
        orders = [np.random.permutation(self.n_outputs) for _ in range (self.n)]
        
        
        self.estimator_chains_ = [singleRC (self.base_estimator, 
                                                 order=orders[_])
                                 for _ in range(self.n)] 
        
        for i in range(self.n):   
                self.estimator_chains_[i].fit(X, Y)
                
        return self

    def predict(self, X):
        
        all_Y_pred = []
        
        for i in range(self.n):
            Y_pred_merged = self.estimator_chains_[i].predict(X)            
            Y_pred_one = Y_pred_merged[:,-self.n_outputs:]
            all_Y_pred.append(Y_pred_one)
        
        all_Y_pred_merged = np.array(all_Y_pred)
        Y_pred = np.mean(all_Y_pred_merged, axis=0)
        
        return Y_pred
  

#################################################################################
############### Multi-Modal Ensemble of Regressor Chains ########################
#################################################################################
    
class EnsembleRegressorChains_MultiModal (BaseEstimator):
    
    def __init__(self, base_estimator, 
                 n=10,
                 sample_size=0.8,
                 clusters=2,
                 random_state=None):
        
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n = n
        self.n_outputs = None
        self.random_state = random_state
        self.clusters = clusters
        
    def _onesample_mean_of_mode(self,x, clusters):
            
        y = x.copy()
            
        cluster_mdl = cluster.KMeans(n_clusters= clusters)
        cluster_mdl.fit(x)
        labels = cluster_mdl.predict(x)
        m = stats.mode(labels)[0][0]
        cleaned = y[np.where(labels == m)]
        z = cleaned.mean(axis=0)
        return z
    

    def fit(self, X, Y, **fit_params):
        
        self.n_outputs = Y.shape[1]
        
        orders = [np.random.permutation(self.n_outputs) for _ in range (self.n)]
        
        
        self.estimator_chains_ = [singleRC_MultiModal (self.base_estimator,
                                                 order=orders[_], 
                                                 sample_size=self.sample_size)
                                 for _ in range(self.n)] 
        
        for i in range(self.n):   
                self.estimator_chains_[i].fit(X, Y)
                
        return self

  
    def predict(self, X):
        
        all_Y_pred = []
        
        for i in range(self.n):
            one_Y_pred = self.estimator_chains_[i].predict(X)            
            all_Y_pred.append(one_Y_pred)
        
        Y_pred = np.empty((0,one_Y_pred.shape[1]), float)
        for j in range(one_Y_pred.shape[0]):
            onesample = np.array([Y[j] for Y in all_Y_pred])
            meanmode_i = self._onesample_mean_of_mode(onesample, self.clusters)
            Y_pred = np.vstack([Y_pred, meanmode_i])

        
        
        return Y_pred    



