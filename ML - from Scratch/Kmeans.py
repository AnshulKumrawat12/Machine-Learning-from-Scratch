# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:00:25 2021

@author: Anshul
"""

import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1 -x2)**2))

class Kmeans:
    def __init__(self, k = 5, n_iters = 100, plot_steps = False ):
        self.k = k
        self.n_iters = n_iters
        self.plot_steps = plot_steps
        
        #store Clusters as list of lists and  Centroids
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        
    
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        #Initialize
        random_sample_idx = np.random.choice(self.n_samples, self.k, replace = False)
        self.centroids = [self.X[i] for i in random_sample_idx]
        
        #Optimize Clusters
        for _ in range(self.n_iters): 
            #Assign samples to closes centroids(create clusters)
            self.clusters = self._create_clusters(self.centroids)
        
            if self.plot_steps:
                self.plot()
        
            #Calculate new centroids from the cluster
            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
        
            #Check if cluster have changed
            if self._is_converged(centroid_old, self.centroids):
                break
        
            if self.plot_steps:
                self.plot()
                
                
                
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
            
        return clusters
    
    def _closest_centroid(self,sample, centroid):
        #distance of current sample to each centroids
        distance = [euclidean_distance(sample, point) for point in centroid]
        idx = np.argmin(distance)
        return idx
    
    def _get_centroids(self, clusters):
        #assign mean value of clusters to centroids
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    
    def _is_converged(self, centroid_old, centroid):
         #distances between each old and new centroid, for all centroids
         distances= [euclidean_distance(centroid_old[i], centroid[i]) for i in range(self.k)]
         return np.sum(distances) == 0
     
    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))
        
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
            
        for point in self.centroids:
            ax.scatter(*point, marker = "X", color = "black", linewidth =2)
            
        plt.show()
        
        
#%%
#Testing

if __name__ == '__main__':
    import sklearn.datasets as datasets
    import numpy as np
        
    
    data = datasets.make_blobs(n_samples=500, n_features = 2, centers =3, shuffle= True, random_state =22)
    X,y = data
    print(X.shape, y.shape)
    
    cluster = len(np.unique(y))
    print(cluster)
    
    kmeans = Kmeans(k = cluster, n_iters=100, plot_steps = True)
    pred = kmeans.predict(X)
    
    