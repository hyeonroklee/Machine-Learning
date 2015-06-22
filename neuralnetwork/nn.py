# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def sigmoid(z):
    return 1.0 /(1.0 + np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_vec = np.vectorize(sigmoid)

class Network():
    def __init__(self,layers):
        self.num_of_layers = len(layers)
        self.thetas = []
        for i in range(1,self.num_of_layers):
            self.thetas.append(np.random.normal(size=(layers[i],layers[i-1]+1)))
        
    def feedforward(self,x):
        x = np.matrix(np.append(x,1.)).T
        m = np.matrix(self.thetas[0])
        x = np.append(sigmoid_vec(m * x),np.ones((m.shape[0],1)),axis=1)
        x = np.matrix(self.thetas[1]) * x.T
        print x
        
    def backprop(self):
        pass
    
if __name__ == '__main__':
    x = np.array( [3,2,1] )
    l = [3,3,1]
    n = Network(l)
    n.feedforward(x)
    
    print sigmoid_vec(x)