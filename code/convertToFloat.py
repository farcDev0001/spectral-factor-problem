import pandas as pd

def saveCsvAsFloat():
    df=pd.read_csv('../input/espec1.csv')
    for column in df.columns:
        df[column]=df[column].apply(lambda num:float(num.replace(',','.')))
    df.to_csv('../input/especNum.csv')

saveCsvAsFloat()