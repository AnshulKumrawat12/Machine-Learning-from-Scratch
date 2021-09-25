# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:00:58 2021

@author: Anshul
"""

import numpy as np
import pandas as pd

# =============================================================================
# #FOUR ways to load the data
# =============================================================================



# =============================================================================
# (1) Using csv library
# =============================================================================

import csv 

FILENAME = "spambase.data"

with open(FILENAME, 'r') as f:
    data = list(csv.reader(f, delimiter=','))

data = np.array(data)
print("Using csv library")
print(data.shape) #(4601,58) 


n_samples, n_features = data.shape
n_features -= 1
X = data[:,0:n_features]
y = data[:,n_features]
print(X.shape, y.shape) # (4601,57) (4601,)



# =============================================================================
# # (2) Using np.loadtxt
# =============================================================================

data = np.loadtxt(FILENAME, delimiter=',')
print("Using np.loadtxt function")
print(data.shape) #(4601,58)

n_samples, n_features = data.shape
n_features -= 1
X = data[:,0:n_features]
y = data[:,n_features]

print(X.shape, y.shape) # (4601,57) (4601,)



# =============================================================================
# # (3) using pd.read_csv
# #     ---- # skiprows = 1, na_values = ["HELLO"]
# #     ---- # df.fillna(0)
# =============================================================================

df = pd.read_csv(FILENAME, delimiter=',', header = None )
print("Using pd.read_csv function")
print(df.shape)

data = df.to_numpy()

n_samples, n_features = data.shape
n_features -= 1
X = data[:,0:n_features]
y = data[:,n_features]
print(X.shape, y.shape) # (4601,57) (4601,)


# =============================================================================
# # (4) Using np.genfromtxt (most preferred)
#   ---- #skip_header = 1, missing_values = "HELLO", filling_values= 0.0, dtype = np.float32,

# =============================================================================

data = np.genfromtxt(FILENAME, delimiter=',')
print("Using np.genfromtxt function (MOST PREFERRED)")
print(data.shape)