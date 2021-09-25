# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 14:36:40 2021

@author: Anshul
"""

import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt

#Dataset loading

data = datasets.make_blobs(n_samples = 150, n_features = 2, centers = 2, cluster_std =1.05, random_state = 2)
X,y = data
print(X.shape, y.shape)

y = np.where(y==0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size= 0.2, random_state=123)
print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape)

#%%

class SVM:
    def __init__(self,lr = 0.01, lambda_param =0.01, n_iter =1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        
        self.weights = None
        self.bias = None
    
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y= np.where(y>0,1,-1)
        
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                    self.bias -= 0
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights -  np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]
                
        
    def predict(self, x):
        y_predicted = np.dot(x, self.weights) -self.bias
        return np.sign(y_predicted)

clf = SVM()
clf.fit(X_train, y_train)
# predicted = svm.predict(X_test)

print(clf.weights, clf.bias) 
       

#%%

def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.weights, clf.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.weights, clf.bias, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.weights, clf.bias, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.weights, clf.bias, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.weights, clf.bias, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.weights, clf.bias, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

visualize_svm()
        