# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""
import numpy as np
import nostalgia as ntg
import matplotlib.pyplot as plt

if __name__ == '__main__':    

    
    '''
    x,y = ntg.simulator.unknown_classification_environment()
    w = ntg.classification.logistic_regression(x,y)
    s1 = np.array([np.min(x[:,0]),np.max(x[:,0])])
    s2 = (-w[0,0]*s1 - w[2,0]) / w[1,0]
    plt.plot(s1,s2) 
    plt.plot(x[:100,0],x[:100,1],'+b')    
    plt.plot(x[100:200,0],x[100:200,1],'+r')
    plt.show()
    '''

    t = [2.,3.]
    x,y = ntg.simulator.unknown_classification_environment()
    ntg.classification.knn(x,y,t)
    
    t = [0,1,1]
    x = [ [1,0,1] , [0,1,0], [1,0,0 ] ]
    y = [ 1,0,0 ]
    ntg.classification.decision_tree(x,y,t)
    
    t = [2.,3.]
    x,y = ntg.simulator.unknown_classification_environment()
    ntg.classification.svm(x,y,t)