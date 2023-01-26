### Learning
from sklearn import multioutput
from sklearn import ensemble
from sklearn import svm
from sklearn import tree


def modelsRegression():
    return [tree.DecisionTreeRegressor(),                    
            ensemble.RandomForestRegressor(n_estimators=10),
            
            multioutput.MultiOutputRegressor(tree.DecisionTreeRegressor()),
            multioutput.MultiOutputRegressor(svm.SVR(kernel='rbf')),
            multioutput.MultiOutputRegressor(ensemble.RandomForestRegressor(n_estimators=10)),
            
            multioutput.RegressorChain(tree.DecisionTreeRegressor()),
            multioutput.RegressorChain(svm.SVR(kernel='rbf')),
            multioutput.RegressorChain(ensemble.RandomForestRegressor(n_estimators=10)) 
            ]


def innerModelsRegression():
    return [tree.DecisionTreeRegressor(),       
            svm.SVR(kernel='rbf'),  
            ensemble.RandomForestRegressor(n_estimators=10)            
            ]



