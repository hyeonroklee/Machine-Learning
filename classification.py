# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:54:41 2015

@author: hyeonrok lee
"""

import numpy as np

def sigmoid(x,w):
    return 1. / (1.+np.exp(-np.dot(x,w)))

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

def knn_classify(dataset,x,k):
    dataSetSize = len(dataSet)
    features = np.array( [ [a[0],a[1]] for a in dataSet ] )
    distMat = np.tile(x,(dataSetSize,1)) - features
    distMat = (np.sum(distMat**2,axis=1))**0.5
    distIdx = distMat.argsort()
    voteCount = {}
    for i in range(k):
        vote = dataSet[distIdx[i]][2]
        voteCount[vote] = voteCount.get(vote,0) + 1
    sortedVoteCount = sorted(voteCount.iteritems(),key=op.itemgetter(1), reverse=True)
    return sortedVoteCount[0][0]