# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:18:53 2021

@author: Anshul
"""

import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from KNN_algo import KNN

cmap = ListedColormap(['#FF0000','#00FF00', '#0000FF'])

iris = datasets.load_iris()
X,y = iris.data, iris.target
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)
print(X_train.shape, y_train.shape)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=y, cmap = cmap, edgecolors='k', s =20 )

clf = KNN(k=5)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

accuracy = np.sum(prediction == y_test) /len(y_test)
print(accuracy)