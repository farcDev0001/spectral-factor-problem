from sklearn.model_selection import cross_val_score
import sklearn.linear_model as lm
from dask.distributed import Client, progress
from sklearn.metrics import accuracy_score
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from dask_ml.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def getLocalDaskCLusterRyzen():
    return Client(n_workers=4, threads_per_worker=4, memory_limit='3.5GB')
    

def getData():
    data= pd.read_csv('../input/especNum.csv')
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='sf'), data.sf,test_size=0.2)
    return X_train, X_test, y_train, y_test

def crossValidation(model,X_train,y_Train,timeCompute=5):
    scores = cross_val_score(model, X_train, y_Train, cv=timeCompute)
    return "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)



def linearReg(X_train, X_test, y_train, y_test):
    model= lm.LinearRegression().fit(X_train,y_train)
    return {'model':model,'name':'LinearRegression','Accuracy':model.score(X_test, y_test)}

def forestReg(X_train, X_test, y_train, y_test):
    model= RandomForestRegressor(max_depth=2, random_state=0,n_estimators=1000).fit(X_train,y_train)
    return {'model':model,'name':'ramdomForestReg','Accuracy':model.score(X_test, y_test)}

def createRamdomForestGrid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    return random_grid

def ramdomSearchCv(model,ramdom_grid,X_train,y_train):
    gridModel= RandomizedSearchCV(estimator = model, param_distributions =ramdom_grid, n_iter = 200, cv = 3, verbose=2, random_state=42)
    gridModel.fit(X_train,y_train)
    return gridModel.best_params_

def ARMRe(X_train, X_test, y_train, y_test):
    model= lm.ARDRegression().fit(X_train,y_train)
    return {'model':model,'name':'LinearRegression','Accuracy':model.score(X_test, y_test)}

def searchModel():
    X_train, X_test, y_train, y_test=getData()
    #functions=[linearReg,modelElasticNetCv,ARMRe]
    #modelNdesc= linearReg(X_train, X_test, y_train, y_test)
    #modelNdesc= linearReg(X_train, X_test, y_train, y_test)
    #modelNdesc= forestReg(X_train, X_test, y_train, y_test)
    modelNdesc=ramdomSearchCv(RandomForestRegressor(),createRamdomForestGrid(),X_train,y_train)
    return modelNdesc

def paralelizeJob():
    client = getLocalDaskCLusterRyzen()
    print(client)
    with joblib.parallel_backend('dask'):
        modelDic=searchModel()
    return modelDic

