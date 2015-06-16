# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def linear_regression_by_normal(x,y):
    t = np.append(x,np.matrix(np.ones(len(x))).T,axis=1)
    y = np.matrix(y)
    w = np.linalg.inv(t.T * t) * (t.T * y) # normal equation
    return w


def linear_regression_by_gradient(x,y):
    # full batch
    t = np.append(x,np.matrix(np.ones(len(x))).T,axis=1)
    w = np.random.normal(size=(t.shape[1],1))
    alpha = 0.0000001 # learning rate
    delta = 0.001
    
    while True:
        delta1 = 0.
        delta2 = 0. 
        for i in range(t.shape[0]):
            delta1 = delta1 + (alpha * ((y[i] - t[i] * w) * t[i,0]))[0,0]
            delta2 = delta2 + (alpha * ((y[i] - t[i] * w) * t[i,1]))[0,0]
        # check if there is a fluctuation to adjust the learning rate
        # print delta1,delta2
        if np.abs(delta1) < delta and np.abs(delta2) < delta:
            break
        w[0] += delta1
        w[1] += delta2
    return w
