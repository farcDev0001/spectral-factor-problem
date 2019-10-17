import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from dask.distributed import Client, progress
import dask.dataframe as dd
from sklearn.model_selection import train_test_split as sklearnSplit
from dask_ml.model_selection import train_test_split as daskSplit
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score


def getLocalDaskCLusterRyzen():
    client= Client(n_workers=4, threads_per_worker=3, memory_limit='8GB')
    print(client)
    return client
    

def getData():
    data= dd.read_csv('../input/especNum.csv').drop(columns='Unnamed: 0')
    X_train, X_test, y_train, y_test = daskSplit(data.drop(columns='sf'), data.sf,test_size=0.4)
    return X_train.compute(), X_test.compute(), y_train.compute(), y_test.compute()


def linearReg(X_train, X_test, y_train, y_test):
    model= lm.LinearRegression().fit(X_train,y_train)
    return {'model':model,'R**2':model.score(X_test,y_test)}
    

def ARMRe(X_train, X_test, y_train, y_test):
    model= lm.ARDRegression().fit(X_train,y_train)
    return {'model':model,'R**2':model.score(X_test, y_test)}

def forestReg(X_train, X_test, y_train, y_test):
    model= RandomForestRegressor(max_depth=2, random_state=0,n_estimators=1000).fit(X_train,y_train)
    return {'model':model,'R**2':model.score(X_test, y_test)}

def createRamdomForestGrid():
    # numero árboles
    n_estimators = [int(x) for x in np.linspace(start = 3, stop = 100, num = 2)]
    # numero features
    max_features = ['auto', 'sqrt']
    # Número máximo de niveles en el árbol
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Número mínimo de muestras requeridas para dividir un nodo
    min_samples_split = [2, 5, 10]
    # Numero mínimo de muestras requeridas para cada nodo
    min_samples_leaf = [1, 2, 4]
    # Metodo de selección de muestras para entrenar cada árbol
    bootstrap = [True, False]

    # Crear el cuadro
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    return random_grid

def searchBestForest(params,X_train, X_test, y_train, y_test,client):
    c=client
    print(c)
    file=open('../output/200a400.txt','w')
    file.write('200 a 400 estimadores\n')
    file.write('\n\n')
    with joblib.parallel_backend('dask'):
        model=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
                        max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=200,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        bestMod={'model':model,'R2_score':r2_score(y_test, y_pred)}
        file.write(str(bestMod)+'\n')
        contador=1
        print(bestMod)
        try:
            for estimators in params['n_estimators']:
                for features in params['max_features']:
                    for dep in params['max_depth']:
                        for samples in params['min_samples_split']:
                            for samplesL in params['min_samples_leaf']:
                                for boot in params['bootstrap']:
                                    model=RandomForestRegressor(bootstrap=boot, criterion='mse', max_depth=dep,
                                    max_features=features, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=samplesL, min_samples_split=samples,
                                    min_weight_fraction_leaf=0.0, n_estimators=estimators,
                                    n_jobs=None, oob_score=False, random_state=None,
                                    verbose=0, warm_start=False)
                                    model.fit(X_train,y_train)
                                    y_pred=model.predict(X_test)
                                    r2=r2_score(y_test, y_pred)
                                    if r2>bestMod['R2_score']:
                                        bestMod={'model':model,'R2_score':r2}
                                        print(bestMod)
                                        file.write(str(bestMod)+'\n')
                                        file.write('\n')
                                        contador+=1
                                    del model
            file.write('numero de modelos en archivo: {}'.format(contador))
            file.close()
        except:
            file.write('numero de modelos en archivo: {}'.format(contador))
            file.close()

def getTheForest():
    count=11
    while(True):
        model = RandomForestRegressor(bootstrap=bool, criterion='mse', max_depth=m,
                      max_features=x, max_leaf_nodes=None,
                      min_impurity_decrease=x, min_impurity_split=None,
                      min_samples_leaf=x, min_samples_split=m,
                      min_weight_fraction_leaf=x, n_estimators=s,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=x, warm_start=False)
        
        data= pd.read_csv('./input/especNum.csv').drop(columns='Unnamed: 0')
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='sf'), data.sf,test_size=0.5)
        model.fit(X_train,y_train)
        y_pred= model.predict(X_test)
        score=r2_score(y_test, y_pred)
        print(score)
        if score > 0.0:
            print(print('YEEEHHHH'))
            dump(model, './output/forest/forestFitted{}.joblib'.format(count))
            count+=1
        del model
            

def paralelizeJobWhithDaskClient(function,client):
    c = client
    print(c)
    with joblib.parallel_backend('dask'):
        function()
        





