# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:01:18 2021

@author: Anshul
"""

import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt


data = datasets.load_iris()
X,y = data.data, data.target
print(X.shape, y.shape)


#%%
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.transformed_vectors = None
        self.mean = None 
        
    def fit(self, X):
        #Mean
        self.mean = np.mean(X)
        X = X - self.mean
        
        #Covariance
        # In X --> 1 row = 1 sample, 1 column = 1 feature
        ## In cov function ---> 1 row = 1 variable(feature), 1 column = 1 sample
        self.cov = np.cov(X.T)
        
        #Eigenvalues, Eigenvectors
        Eigenvalues, Eigenvectors  = np.linalg.eig(self.cov)
        
        #Sort
        # 1 column = 1 eigenvector
        # eigenvalues --> column vector
        Eigenvectors = Eigenvectors.T
        idx =  np.argsort(Eigenvalues.T)[::-1]
        Eigenvalues = Eigenvalues[idx]
        Eigenvectors = Eigenvectors[idx]
        
        #transform
        self.transformed_vectors = Eigenvectors[0: self.n_components].T
    
    def transform(self,X):
        X = X - self.mean
        return np.dot(X, self.transformed_vectors)
        
    
#%%

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)


print(X.shape)
print(X_projected.shape)



#%%

x1 = X_projected[:,0]
x2 = X_projected[:,1]

plt.scatter(x1,x2, c = y, edgecolor ='none', alpha = 0.8, cmap = plt.cm.get_cmap('viridis',3))

plt.xlabel('Principal Component 1')
plt.ylabel('Principa; Component 2')
plt.colorbar()
plt.show()

