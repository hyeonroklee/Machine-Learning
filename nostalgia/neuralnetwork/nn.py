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
        x = np.matrix(np.append(np.array(x),1.)).T
        for i in range(self.num_of_layers-1):
            active = sigmoid_vec(np.matrix(self.thetas[i]) * x) 
            x = np.matrix(np.append(active.tolist(),1)).T
        return active
        
    def backprop(self):
        pass
    
if __name__ == '__main__':
    x = np.array([3,2,1])
    layers = [3,3,1]
    n = Network(layers)
    print n.feedforward(x)