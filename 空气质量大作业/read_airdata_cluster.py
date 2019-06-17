# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:12:41 2019

@author: 李奇
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from AE_model import AE

from sklearn.preprocessing import MinMaxScaler#,Normalizer,StandardScaler

from sklearn.manifold  import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA


from sklearn import svm


#%%
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
    datapart=data.loc[:,['PM25','PM10','SO2','CO','NO2','O3_8h'] ]
    outputdata=np.array(datapart)

    scaler = MinMaxScaler(feature_range=(0, 1))
    outputdata = scaler.fit_transform(outputdata)
    print("==========预处理完毕=======")
    
    return outputdata

def dimension2(data,method='AE'):
    '''
    通过不同的方法将数据降到2维方便可视化
    '''
    print("==========开始降维=========")
    print('降维方法为',method)
    if method=='TSNE':
        data=TSNE(n_components=2, n_iter=300).fit_transform(data)
    if method=='PCA':
        data = PCA(n_components=2).fit_transform(data)
    if method=='KernelPCA': #kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
        data=KernelPCA(n_components=2, kernel='linear').fit_transform(data)
    if method=='SparsePCA':
        data=SparsePCA(n_components=2).fit_transform(data)
    if method=='AE':
        data=AE(n_components=2).fit_transform(data)
    print("==========降维完毕=========")
    return data

def cluster(X,method='KMeans'):
    '''
    不同的聚类方法
    '''
    print("==========开始聚类=========")
    print('聚类方法为:',method)
    
    if method=='DBSCAN':
        y_pred = DBSCAN(eps = 0.9).fit_predict(X)#通过改变DBSCAN的参数
        
    if method=='KMeans':
        y_pred=KMeans(n_clusters=6).fit_predict(X)
        
    if method=='SpectralClustering':
        y_pred=SpectralClustering().fit_predict(X)
        
    if method=='AP':
        y_pred=AffinityPropagation(0.6).fit(X).labels_
        
    plot_data(X,y_pred)
    
    print("==========聚类完毕=========")
    return y_pred

def plot_data(X,y_pred):    
    '''
    画图
    '''
    color=['gold','darkorange','aqua','violet','deepskyblue','lime']
    
    for i in range(y_pred.max()+1):
        
        index = np.where(y_pred==i)[0]
        point=X[index]
        plt.scatter(point[:, 0], point[:, 1], c=color[i],label='air condition'+str(i))
        print(len(index))
        
    plt.legend()
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.savefig('cluster_result'+'.svg')
    plt.savefig('cluster_result'+'.png')
    plt.show()   
    
    return None


def classify(X,Y,method='svm'):
    '''
    分类器
    '''
    print("=======开始训练分类器======")
    print('采用的分类器为',method)
    if method=='svm':
        
        clf = svm.SVC(gamma='auto')
        clf.fit(X, Y)
    print("==========训练完毕=========")
    return clf
#%%    
if __name__=='__main__':
    
    airdata=transfer_to_np()
    
    airdata2=dimension2(airdata)
    
    label=cluster(airdata2)
    
    classifier=classify(airdata2,label)
    
    predition=classifier.predict(airdata2)
    
    acc=(predition==label).sum()/len(label)
    
    print('clf acc is %4f'%acc)
    
    