# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""
import numpy as np
import nostalgia as ntg
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y = ntg.simulator.unknown_regression_environment()
    w1 = ntg.regression.linear_regression_by_equation(x,y)
    w2 = ntg.regression.linear_regression_by_gradient(x,y)
    y1 = np.append(x,np.matrix(np.ones(len(x))).T,axis=1) * w1 
    y2 = np.append(x,np.matrix(np.ones(len(x))).T,axis=1) * w2
    plt.plot(x,y,'+r')
    plt.plot(x,y1,'-b')    
    plt.plot(x,y2,'-g')
    plt.show()
    print w1
    print w2
    print w1-w2
    