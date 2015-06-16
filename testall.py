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
    return x,t.dot(p) + np.random.normal(loc=0.0,scale=10.0,size=(x.shape[0],p.shape[1]))

def unknown_classification_environment():
    pass

if __name__ == '__main__':
    x,y = unknown_regression_environment()
    w1 = reg.linear_regression_by_normal(x,y)
    #w2 = reg.linear_regression_by_batch(x,y)
    plt.plot(x[:,0],y,'+r')
    plt.plot(x,x*w1)
    #plt.plot(x,x*w1)
    plt.show()
    
'''
    x = np.matrix( [ np.linspace(0,100,101), np.ones(101) ] ).T.tolist()
    p = [ [2.] , [3.] ]
    y = linear_model(x,p)
    plt.plot(x,y,'+r')
'''    
    
'''
    x = [ [t,1.] for t in np.linspace(0,100,101) ]
    # p = [ [2.],[3.] ]
    y = unknown_environment(x)
    plt.plot(np.array(x)[:,0],y,'+r')
'''
    
'''
x = [ [x,1.] for x in np.linspace(0,10,100)]
x = np.array(x)
y = linear_model(x)
w,w1 = linear_regression(x,y)

print w,w1
 
plt.plot(x[:,0],y,'+r')
plt.plot(x[:,0],x*w)
plt.plot(x[:,0],np.matrix(x) * np.matrix(w1))
'''