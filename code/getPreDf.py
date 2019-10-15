import pandas as pd
from sklearn.metrics import r2_score

def errorAbsoluto(row):
    error = row.sf-row.pred
    if error < 0:
        error=error*(-1)
    return error

def getDf(y_test,y_pred):
    df=pd.DataFrame(y_test)
    df['pred']=y_pred
    df['errorAbs']=df.apply(lambda row:errorAbsoluto(row),axis=1)
    df=df.reset_index().drop(columns='index')
    df=df.reset_index()
    print('Valores predecidos correspondientes al 40% de los datos:',df.shape[0],
      'r2_score:',r2_score(df.sf, df.pred))
    return df