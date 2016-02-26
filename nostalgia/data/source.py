
import struct
import os
import numpy as np

def generate_linear_regression_data(n=20,low_bound=-10,high_bound=10,noize=1,f = lambda x : x + 2):
    x = np.array([ np.random.uniform(low_bound,high_bound,n) ])
    y = f(x) + np.random.normal(0,noize,n)
    return x.T,y.T

def generate_linear_classification_data(n1=10,n2=10):
    mean1 = [1.,1.]
    cov1 = [ [ 1., 0.5 ] , [ 0.5, 1. ] ]
    d1 = np.random.multivariate_normal(mean1,cov1,n1)
    mean2 = [3.,3.]
    cov2 = [ [ 1., 0.5 ] , [ 0.5, 1. ] ]
    d2 = np.random.multivariate_normal(mean2,cov2,n2)
    x = np.append(d1.T,d2.T,axis=1)
    y = np.append([np.zeros(n1)],[np.ones(n2)],axis=1)
    return x.T,y.T

def generate_random_walk(n=100,mu=[0.,0.],sigma=[[1.,0.],[0.,1.]]):
    num_of_features = len(mu)
    series = np.array([[1.] * num_of_features])
    for i in range(1,n):
        series = np.append(series,[series[i-1,:]+series[i-1,:]*np.random.multivariate_normal(mu,sigma)],axis=0)
    return series

def read_mnist_digit():
    train_x = []
    train_y = []
    test_x  = []
    test_y  = []

    current_dir = os.path.dirname(__file__)

    f = open(current_dir + '/train-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        nrows = struct.unpack('>I',f.read(4))[0]
        ncols = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            data = struct.unpack('784B', f.read(nrows*ncols))
            train_x.append(data)
    finally:
        f.close()
    
    f = open(current_dir + '/train-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            label = struct.unpack('B',f.read(1))[0]
            train_y.append([label])
    finally:
        f.close()
        
    f = open(current_dir + '/t10k-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        nrows = struct.unpack('>I',f.read(4))[0]
        ncols = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            data = struct.unpack('784B', f.read(nrows*ncols) )
            test_x.append(data)
    finally:
        f.close()
    
    f = open(current_dir + '/t10k-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            label = struct.unpack('B',f.read(1))[0]
            test_y.append([label])
    finally:
        f.close()        
        
    return train_x,train_y,test_x,test_y