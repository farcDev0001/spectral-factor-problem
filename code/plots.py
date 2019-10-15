import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def getNsaveRandomPredPlot(numeroMuestras,dfPred,pathOutput):
    dfs=dfPred.sample(n=numeroMuestras, random_state=None).reset_index()

    plt.figure(figsize=(35,22))

    plt.scatter(dfs.index, dfs.sf, c='red')
    plt.scatter(dfs.index, dfs.pred, c='blue')
    plt.plot(dfs.index, dfs.errorAbs, c='green')

    plt.yticks([i/10 for i in range(12)])
    plt.xticks(dfs.index)
    plt.xlabel('Número de muestra',fontsize=20)
    plt.ylabel('valorSf',fontsize=20)
    plt.title('{} muestras aleatorias entre las {} predicciones. Valor de r2_score: {}'.format(numeroMuestras,dfs.shape[0],r2_score(dfs.sf, dfs.pred)),fontsize=20)
    plt.legend(['Error Absoluto Positivo','Real','Prediccion'],fontsize=18)

    plt.grid()

    plt.savefig(pathOutput)