import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def getNsaveRandomPredPlot(numeroMuestras,dfPred,pathOutput):
    dfs=df.sample(n=numeroMuestras, random_state=None).reset_index()

    plt.figure(figsize=(35,22))

    plt.scatter(dfs.index, dfPred.sf, c='red')
    plt.scatter(dfs.index, dfPred.pred, c='blue')
    plt.plot(dfs.index, dfs.errorAbs, c='green')

    plt.yticks([i/10 for i in range(12)])
    plt.xticks(dfPred.index)
    plt.xlabel('Número de muestra',fontsize=20)
    plt.ylabel('valorSf',fontsize=20)
    plt.title('{} muestras aleatorias entre las {} predicciones. Valor de r2_score: {}'.format(numero_muestras,df.shape[0],r2_score(df.sf, df.pred)),fontsize=20)
    plt.legend(['Error Absoluto Positivo','Real','Prediccion'],fontsize=18)

    plt.grid()

    plt.savefig(pathOutput)