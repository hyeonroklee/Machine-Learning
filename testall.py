# -*- coding: utf-8 -*-

"""

@author: hyeonrok lee

"""

import numpy as np
import matplotlib.pyplot as plt
import regression as reg
import classification as cls

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

if __name__ == '__main__':
    '''
    
    x,y = unknown_regression_environment()
    w1 = reg.linear_regression_by_equation(x,y)
    w2 = reg.linear_regression_by_gradient(x,y)
    y1 = np.append(x,np.matrix(np.ones(len(x))).T,axis=1) * w1 
    y2 = np.append(x,np.matrix(np.ones(len(x))).T,axis=1) * w2
    plt.plot(x,y,'+r')
    plt.plot(x,y1,'-b')    
    plt.plot(x,y2,'-g')
    plt.show()
    print w1
    print w2
    print w1-w2
    '''
    
    '''
    x,y = unknown_classification_environment()
    w = cls.logistic_regression(x,y)
    s1 = np.array([np.min(x[:,0]),np.max(x[:,0])])
    s2 = (-w[0,0]*s1 - w[2,0]) / w[1,0]
    plt.plot(s1,s2) 
    plt.plot(x[:100,0],x[:100,1],'+b')    
    plt.plot(x[100:200,0],x[100:200,1],'+r')
    plt.show()
    '''
    
    t = [2.,3.]
    x,y = unknown_classification_environment()
    cls.knn(x,y,t)
    
    