# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:33:45 2021

@author: Anshul
"""


import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Load Dataset
dataset = datasets.make_regression(n_samples=100, n_features =1, noise =20, random_state=4)
X,y = dataset
print(X.shape, y.shape)

#Split data into train-test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,  random_state=1234)
print(X_train.shape,y_train.shape)


class LinearRegression :
    def __init__(self, lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        
    def fit(self,X,y):
        #Initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        
        #Gradient calculation
        for _ in range(self.n_iter):
            y_hat = np.dot(X, self.weights) + self.bias
            #Gradient calculation
            dw = (1/n_samples) * np.dot(X.T, (y_hat-y))
            db = (1/n_samples) * np.sum(y_hat-y)
            
            
            #Update rule
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    
    def predict(self,x):
        y_predicted = np.dot(x, self.weights) + self.bias
        return y_predicted
    

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)


def mse(y, y_pred):
    return np.mean((y - y_pred)**2)
        
mse_value = mse(y_test, y_predict)
print(f'MSE value : {mse_value:}')


y_pred_line = regressor.predict(X_test)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize = (8,6))
m1 = plt.scatter(X_train,y_train, color = cmap(0.9), s =10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s=10)
plt.plot(X_test,y_pred_line, color ='black', linewidth =2, label="Prediction")
plt.show()

