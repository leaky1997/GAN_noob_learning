# -*- coding: utf-8 -*- 
"""
Created on Mon Jun 10 19:49:25 2019

@author: 李奇
"""
from math import sqrt
import numpy as np

import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import l2
from keras.layers import Dropout
import pandas as pd

def read_csv_file(f, logging=False):
    '''
    读取csv文件
    '''
    print("==========读取数据=========")  
    data =  pd.read_csv(f,encoding='gbk')                                      #gbk
    if logging:  
        print(data.head(5))  
        print(f, "包含以下列")  
        print(data.columns.values)  
        print(data.describe())  
        print(data.info()) 
    print("==========读取完毕=========")
    return data  


def transfer_to_np(filename = 'soochow.csv'):
    '''
    将dataframe转换成array格式并归一化
    '''
    print("==========数据预处理=======")
    data = read_csv_file(filename)
    datapart=data.loc[:,['AQI','PM25','PM10','SO2','CO','NO2','O3_8h'] ]
    outputdata=np.array(datapart)
    outputdata.astype('float64')
    scaler = MinMaxScaler(feature_range=(0, 1))###不归一化
    outputdata = scaler.fit_transform(outputdata)

    print("==========预处理完毕=======")
    
    return outputdata,scaler

def create_dataset(dataset,look_back=600):
    '''
    前600天预测30天
    
    '''
    dataX, dataY=[], []

    for i in range(len(dataset)-look_back-30):

        a=dataset[i:(i+look_back)]

        dataX.append(a)

        dataY.append(dataset[i+look_back:i+look_back+30][:,0])

    return np.array(dataX), np.array(dataY)

def Model():
    model = Sequential()
    model.add(LSTM(32, 
                   input_shape=(600, 7),
                   dropout=0.2,
                   return_sequences=True))

    model.add(LSTM(32,
                   dropout=0.2))

    model.add(Dense(30))
    model.compile(loss='mae', optimizer='adam')
    return  model


#%%
    
if __name__=='__main__':
    
# load dataset
    

    data_pre,scaler=transfer_to_np()
       
    data,ydata=create_dataset(data_pre)
    
    data=data.reshape([-1,600,7])
    
    
    
    train_X, train_y = data[:1300], ydata[:1300]
    
    test_X, test_y = data[-85:], ydata[-85:]

    train_X = train_X.reshape((train_X.shape[0],  train_X.shape[1],7))
    test_X = test_X.reshape((test_X.shape[0],  test_X.shape[1],7))

     
    # design network
    model = Model()
    # fit network
    history = model.fit(train_X, train_y, epochs=4000, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=True)
    # evaluate the model
    scores = model.evaluate(test_X, test_y)
    #print scores
    #lcd print("\n\n\t%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    #%% plot history
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    # make a prediction
    y_pred = model.predict(test_X)
    
    
    # invert scaling for forecast

    # invert scaling for actual
    test_y = test_y
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_pred, test_y))
    scaler.min_=scaler.min_[0]#######将scaler从7维改到AQI那一维
    scaler.scale_=scaler.scale_[0]
    y_pred=scaler.inverse_transform(y_pred)
    test_y = scaler.inverse_transform(test_y)
    print('Test RMSE: %.3f' % rmse)
    
    
    plt.plot(test_y[:,0], label='actual')
    
    plt.plot(y_pred[:,0], label='forecast')
    plt.legend()
    plt.savefig('predition.svg')
    plt.show()
    
#    for i in range(30):
#        y_pred = model.predict(test_X)
#        test_X.append(y_pred[-1])
#        test_X=test_X[-30:]
#    test_X=scaler.inverse_transform(test_X)
#    plt.plot(test_X[:0]label='forecast')

#    train_X, train_y = data[:1913], ydata[:1913]
#    y_predtrain = model.predict(train_X)
#    rmse = sqrt(mean_squared_error(y_predtrain, train_y))
#    y_predtrain=scaler.inverse_transform(y_predtrain)
#    train_y = scaler.inverse_transform(train_y)
#    
#    plt.plot(train_y[:,0][500:700], label='trainactual')
#    
#    plt.plot(y_predtrain[:,0][500:700], label='trainforecast')
#    plt.legend()
#    plt.show()
