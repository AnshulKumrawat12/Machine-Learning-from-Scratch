# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:19:20 2021

@author: Anshul
"""

import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt


data = datasets.load_iris()  # 150 x 4
X,y = data.data, data.target
print(X.shape, y.shape[0])

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminant = None
    
    def fit(self, X,y):
        self.n_samples, self.n_features = X.shape 
        class_labels = np.unique(y)
        self.overall_mean = np.mean(X, axis = 0)
        
        #Initialize S_W, and S_B
        S_W = np.zeros((self.n_features, self.n_features)) # 4x4
        S_B = np.zeros((self.n_features, self.n_features)) # 4x4
        
        for c in class_labels:
            X_c = X[y==c]
            c_mean = np.mean(X_c, axis = 0)
            
            # (4,150) * (150, 4) = (4,4)
            S_W += (X_c - c_mean).T.dot(X_c - c_mean)
            
            #S_B
            n_c = X_c.shape[0]
            mean_diff = (c_mean - self.overall_mean).reshape(self.n_features,1) # (4,) --> (4x1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        A = np.linalg.inv(S_W).dot(S_B)
    
        Eigenvalues, Eigenvectors = np.linalg.eig(A)
        Eigenvectors = Eigenvectors.T
        
        idx = np.argsort(abs(Eigenvalues))[::-1]
        
        Eigenvalues = Eigenvalues[idx]
        Eigenvectors = Eigenvectors[idx]
        
        self.linear_discriminant = Eigenvectors[0:self.n_components] 
        
    def transform(self, X):
        return np.dot(X, self.linear_discriminant.T)
    
    

lda = LDA(2)
lda.fit(X,y)
X_projected = lda.transform(X)

print(X.shape, X_projected.shape)

x1, x2 = X_projected[:, 0], X_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.colorbar()
plt.show()