# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def linear_model(x,p):
    x = np.array(x)
    p = np.array(p)
    return x.dot(p) + np.random.normal(loc=0.0,scale=1.0,size=(x.shape[0],p.shape[1]))
    
def linear_regression(x,y):
    x = np.matrix(x)
    y = np.matrix(y)
    w = np.linalg.inv(x.T * x) * (x.T * y) # normal equation
    return w
    
def unknown_environment(x):
    x = np.array( [ t + [1.] for t in x] )
    p = np.array( [[2.],[3.]] )
    return x.dot(p) + np.random.normal(loc=0.0,scale=10.0,size=(x.shape[0],p.shape[1]))
    
if __name__ == '__main__':
    x = np.matrix( [ np.linspace(0,100,101), np.ones(101) ] ).T.tolist()
    p = [ [2.] , [3.] ]
    y = linear_model(x,p)
    plt.plot(x,y,'+r')
    
    
    
'''
    x = [ [t,1.] for t in np.linspace(0,100,101) ]
    # p = [ [2.],[3.] ]
    y = unknown_environment(x)
    plt.plot(np.array(x)[:,0],y,'+r')
'''
'''
    
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