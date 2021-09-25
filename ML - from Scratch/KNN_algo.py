# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:34:23 2021

@author: Anshul
"""

import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        prediction_labels = [self._prediction(x) for x in X]
        return np.array(prediction_labels)
        
    #Helper function to calculate distance for each point
    def _prediction(self,x):
        #compute distance
        distance = [euclidean_distance(x,x_train) for x_train in self.X_train]
        
        # get k nearest neighbours, labels
        point_index = np.argsort(distance)[:self.k]  # It gives the indexes of top k nearest neighbours
        point_label = [self.y_train[i] for i in point_index]
        
        #majority vote, most common labels
        predict = Counter(point_label).most_common(1)
        return predict[0][0]
        
        
        
        
# =============================================================================
# from collections import Counter
# x = [1,2,3,1,1,2,3,2,3,2,3,2,3,2]
# p = Counter(x).most_common(1)
# print(p) #Print [(2,6)]
# print(p[0]) #Print (2,6)
# print(p[0][0]) #Print(2) 
# =============================================================================


        
        