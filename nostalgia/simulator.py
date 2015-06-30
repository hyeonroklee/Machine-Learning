# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def unknown_regression_environment(): 
    x = np.matrix(np.linspace(0,100,101)).T    
    t = np.append(x,np.matrix(np.ones(len(x))).T,axis=1)
    p = np.array( [[2.],[3.]] )
    return x,t.dot(p) + np.random.normal(loc=0.0,scale=10.,size=(t.shape[0],p.shape[1]))

def unknown_classification_environment():
    n1 = 100        
    mean1 = [1.,1.]
    cov1 = [ [ 1., 0.5 ] , [ 0.5, 1. ] ]    
    d1 = np.random.multivariate_normal(mean1,cov1,n1)

    n2 = 100        
    mean2 = [3.,3.]
    cov2 = [ [ 1., 0.5 ] , [ 0.5, 1. ] ]    
    d2 = np.random.multivariate_normal(mean2,cov2,n2)
    
    x = np.append(d1,d2,axis=0)
    y = np.append(np.matrix(np.zeros(n1)).T,np.matrix(np.ones(n2)).T,axis=0)
    return x,y