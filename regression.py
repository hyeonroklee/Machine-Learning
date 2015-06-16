# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def linear_regression_by_normal(x,y):
    x = np.matrix(x)
    y = np.matrix(y)
    w = np.linalg.inv(x.T * x) * (x.T * y) # normal equation
    return w


def linear_regression_by_batch(x,y):
    # full batch
    w1 = np.random.normal(size=(x.shape[1],1))        
    alpha = 0.001 # learning rate
   
    for ii in range(300):
        delta1 = [0.]
        delta2 = [0.] 
        for i in range(60):
            #print y[i], x[i] * w1, x[i,0]
            delta1 = delta1 + alpha * ((y[i] - x[i] * w1) * x[i,0])
            delta2 = delta2 + alpha * ((y[i] - x[i] * w1) * x[i,1])
        w1[0] += delta1[0,0]
        w1[1] += delta2[0,0]
    return w,w1
