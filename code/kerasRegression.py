import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def getScaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data.drop(columns='sf'))
    return scaler

def fitNgetScaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data.drop(columns='sf'))
    return scaler

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def getData():
    data= pd.read_csv('./input/especNum.csv').drop(columns='Unnamed: 0')
    train_dataset = data.sample(frac=0.8,random_state=None)
    test_dataset = data.drop(train_dataset.index)
    train_labels = train_dataset.pop('sf')
    test_labels = test_dataset.pop('sf')
    return train_dataset, test_dataset, train_labels ,test_labels

def build_model(train_dataset):
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1,activation='linear')])

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
    #optimizer= tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    #optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)perce no valer
    #optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    
    
    model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])
    return model

def eternal():
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    EPOCHS = 1000
    score=0.0
    count=1
    scaler=getScaler(pd.read_csv('./input/especNum.csv').drop(columns='Unnamed: 0'))
    while(True):
        train_dataset, test_dataset, train_labels ,test_labels=getData()
        model = build_model(train_dataset)
        train_dataset=scaler.transform(train_dataset)
        test_dataset=scaler.transform(test_dataset)
        model.fit(train_dataset, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
        y_pred= model.predict(test_dataset).flatten()
        newScore=r2_score(test_labels, y_pred)
        print(newScore)
        if newScore>score:
            print('YEEEHHHHHHHHHHHHHHHHHHHHH')
            score=newScore
            model.save('./output/keras/kerasSGDScaler{}.h5'.format(count))
            count+=1
        elif score==1:
            raise Exception("APOCALYPSE!!!!!!!")
        del model



