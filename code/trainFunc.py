from sklearn.model_selection import cross_val_score
import sklearn.linear_model as lm
from dask.distributed import Client, progress
from sklearn.metrics import accuracy_score
import dask.dataframe as dd
from sklearn.model_selection import train_test_split as sklearnSplit
from dask_ml.model_selection import train_test_split as daskSplit
import dask_ml.model_selection as dcv
from scipy.stats import expon
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
from sklearn.metrics import mean_absolute_error


def getLocalDaskCLusterRyzen():
    return Client(n_workers=4, threads_per_worker=4, memory_limit='3.5GB')
    

def getData():
    data= dd.read_csv('../input/especNum.csv')
    X_train, X_test, y_train, y_test = daskSplit(data.drop(columns='sf'), data.sf,test_size=0.2)
    return X_train.compute(), X_test.compute(), y_train.compute(), y_test.compute()

'''def crossValidation(model,X_train,y_Train,timeCompute=5):
    scores = cross_val_score(model, X_train, y_Train, cv=timeCompute)
    return "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)'''

def linearReg(X_train, X_test, y_train, y_test):
    model= lm.LinearRegression().fit(X_train,y_train)
    return {'model':model,'R**2':model.score(X_test,y_test)}
    




def ARMRe(X_train, X_test, y_train, y_test):
    model= lm.ARDRegression().fit(X_train,y_train)
    return {'model':model,'name':'LinearRegression','Accuracy':model.score(X_test, y_test)}

def searchModel():
    X_train, X_test, y_train, y_test=getData()
    #functions=[linearReg,modelElasticNetCv,ARMRe]
    modelNdesc= linearReg(X_train, X_test, y_train, y_test)
    #modelNdesc= linearReg(X_train, X_test, y_train, y_test)
    #modelNdesc= forestReg(X_train, X_test, y_train, y_test)
    #modelNdesc=ramdomSearchCvpandas(RandomForestRegressor(),createRamdomForestGrid(),X_train,y_train)
    return modelNdesc

def paralelizeJob():
    client = getLocalDaskCLusterRyzen()
    print(client)
    with joblib.parallel_backend('dask'):
        
        return searchModel()

def job():
    modelNdesc=searchModel()
    return modelNdesc



