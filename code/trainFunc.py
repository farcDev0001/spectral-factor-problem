from sklearn.model_selection import cross_val_score
import sklearn.linear_model as lm
from dask.distributed import Client, progress
from sklearn.metrics import accuracy_score
#import dask.dataframe as dd
from sklearn.model_selection import train_test_split
#from dask_ml.model_selection import train_test_split
import pandas as pd
import joblib

def getLocalDaskCLusterRyzen():
    return Client(n_workers=4, threads_per_worker=4, memory_limit='3.5GB')
    

def getcarsData():
    cars= pd.read_csv('../output/dfForTrain1.csv').drop(columns='Unnamed: 0')
    X_train, X_test, y_train, y_test = train_test_split(cars.drop(columns='price'), cars.price,test_size=0.1)
    return X_train, X_test, y_train, y_test

def crossValidation(model,X_train,y_Train,timeCompute=5):
    scores = cross_val_score(model, X_train, y_Train, cv=timeCompute)
    return "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)



def linearReg(X_train, X_test, y_train, y_test):
    model= lm.LinearRegression().fit(X_train,y_train)
    return {'model':model,'name':'LinearRegression','Accuracy':model.score(X_test, y_test)}

def ARMRe(X_train, X_test, y_train, y_test):
    model= lm.ARDRegression().fit(X_train,y_train)
    return {'model':model,'name':'LinearRegression','Accuracy':model.score(X_test, y_test)}

def searchModel():
    X_train, X_test, y_train, y_test=getcarsData()
    #functions=[linearReg,modelElasticNetCv,ARMRe]
    #modelNdesc= linearReg(X_train, X_test, y_train, y_test)
    modelNdesc= linearReg(X_train, X_test, y_train, y_test)
    return modelNdesc

def paralelizeJob():
    client = getLocalDaskCLusterRyzen()
    print(client)
    with joblib.parallel_backend('dask'):
        modelDic=searchModel()
    return modelDic

