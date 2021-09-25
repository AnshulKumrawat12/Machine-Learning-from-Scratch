# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 13:37:39 2021

@author: Anshul
"""


import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = datasets.make_blobs(n_samples=150, n_features = 2, centers=2 , cluster_std=1.05, random_state=2 )
X,y = data
print(X.shape, y.shape)

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#%%

class Perceptron:
    def __init__(self, lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
            
        self.weights = None
        self.bias = None
        self.activation_func = self._step_func
        
    def fit(self, X,y):
        n_samples, n_features = X.shape
        
        #Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y = np.array([1 if i>0 else 0 for i in y])

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                y_pred = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(y_pred) #Applied on single value x_i
                
                #updation
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update * 1

        
    def _step_func(self, x):
        return np.where(x>=0,1,0) #where used for calculating for each vector(multiple values)
        
        
    def predict(self,x):
        y_prediction = np.dot(x,self.weights) + self.bias
        y_prediction = self.activation_func(y_prediction) # Applied on multiple values of x
        return y_prediction
        

#%%

percep = Perceptron(lr = 0.01, n_iter = 1000)
percep.fit(X_train,y_train)
prediction = percep.predict(X_test)

def accuracy(y_pred,y):
    return (y_pred == y).sum() / len(y)

print("Accuracy of model:",accuracy(prediction,y_test))


#%%
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:, 0], X_train[:,1], marker = 'o', c = y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-percep.weights[0] * x0_1 - percep.bias) / percep.weights[1]
x1_2 = (-percep.weights[0] * x0_2 - percep.bias) / percep.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()
