# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def sigmoid(x,w):
    return 1. / (1.+np.exp(-np.dot(x,w)))
    
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def logistic_regression(x,y):
    # full batch
    t = np.append(x,np.matrix(np.ones(len(x))).T,axis=1)
    w = np.random.normal(size=(t.shape[1],1))
    alpha = 0.001 # learning rate
    delta = 0.01
    
    while True:
        delta1 = 0.
        delta2 = 0.
        delta3 = 0.
        for i in range(t.shape[0]):
            delta1 = delta1 + (alpha * ((y[i] - sigmoid(t[i],w)) * t[i,0]))[0,0]
            delta2 = delta2 + (alpha * ((y[i] - sigmoid(t[i],w)) * t[i,1]))[0,0]
            delta3 = delta3 + (alpha * ((y[i] - sigmoid(t[i],w)) * t[i,2]))[0,0]
        # check if there is a convergence to adjust the learning rate
        # print delta1,delta2,delta3
        if np.abs(delta1) < delta and np.abs(delta2) < delta and np.abs(delta3) < delta:
            break
        w[0] += delta1
        w[1] += delta2
        w[2] += delta3                
    return w

def knn(x,y,t,k=10):
    d = np.sqrt(np.sum( (np.array(x) - np.array(t)) * (np.array(x) - np.array(t)),axis=1) )
    idx = d.argsort()
    votes = {}
    for i in range(k):
        v = y[idx[i],0]
        votes[v] = votes.get(v,0) + 1
    print votes