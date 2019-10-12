from sklearn.decomposition import PCA
import pandas as pd

def getPCAoneCom(pdcolumn):
    columns = pd.get_dummies(pdcolumn)
    pca = PCA(n_components=1)
    pca.fit(columns)
    return pca.transform(columns).reshape(1, -1)[0]

def convertToNumber():
    cars=pd.read_csv('../output/partialCleanOnlyM.csv').drop(columns=['Unnamed: 0','make'])
    
    for col in list(cars.columns)[:-1]:
        cars[col]=getPCAoneCom(cars[col]).astype('float32')
        #cars[col]=cars[col].apply(lambda p:p/10)
        print(col,'converted and normalized')
    #cars.price = cars.price.apply(lambda p:p/10**10)#normalicing prize
    cars.to_csv('../output/dfForTrain1.csv')

from sklearn.model_selection import cross_val_score
def crossValidation(model,X_train,y_Train,timeCompute=5):
    scores = cross_val_score(model, X_train, y_Train, cv=timeCompute)
    return "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

from sklearn.model_selection import RandomizedSearchCV
def ramdomSearchCvpandas(model,ramdom_grid,X_train,y_train):
    gridModel= RandomizedSearchCV(estimator = model, param_distributions =ramdom_grid, n_iter = 100, 
    cv = 3, verbose=2, random_state=42, n_jobs = 8)
    gridModel.fit(X_train,y_train)
    return gridModel.best_params_

def ramdomSearchCvDask(model,ramdom_grid,X_train,y_train):
    gridModel= dcv.RandomizedSearchCV(model, ramdom_grid, n_iter=100)
    gridModel.fit(X_train,y_train)
    return gridModel.best_params_