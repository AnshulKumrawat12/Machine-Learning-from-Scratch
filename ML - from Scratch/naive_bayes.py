# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:35:21 2021

@author: Anshul
"""

import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

dataset = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1230)
X,y = dataset
print(X.shape, y.shape)

X_train, X_test, y_train,y_test  = train_test_split(X,y, test_size= 0.2, random_state=10)
print(X_train.shape,y_train.shape) 

class NaiveBayes:
    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.n_classes = np.unique(y)
        _classes = len(self.n_classes)
        
        #Init mean,var,prior
        self._mean = np.zeros((_classes, n_features), dtype=np.float64)
        self._var = np.zeros((_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(_classes, dtype = np.float64)
        
        for c in self.n_classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._prior[c] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    
    def _predict(self, x):
        posteriors =[]
        
        for idx, c in enumerate(self.n_classes):
            prior = np.log(self._prior[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior_prob = prior + class_conditional
            
            posteriors.append(posterior_prob)
        
        return self.n_classes[np.argmax(posteriors)]
        
    
    def _pdf(self,class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        
        numerator = np.exp(-(x - mean)**2/ 2 * var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator        
            
    
classifier = NaiveBayes()
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)

accuracy = np.sum(prediction == y_test) / len(y_test)

print("Accuracy:" , accuracy)