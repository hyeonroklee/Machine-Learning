
import struct
import numpy as np

def generate_random_walk(n=100,mu=None,sigma=None):
    num_of_features = len(mu)
    series = np.array([[1.] * num_of_features])
    for i in range(1,n):
        r = [ series[i-1,:] + series[i-1,:] * np.random.multivariate_normal(mu,sigma) ]
        series = np.append(series,r,axis=0)
    return np.array(series)

def read_mnist_digit_train():
    train = []
    test = []
    nrows = 0
    ncols = 0
    
    f = open('train-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        nrows = struct.unpack('>I',f.read(4))[0]
        ncols = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            data = struct.unpack('784B', f.read(nrows*ncols) )
            train.append([ data ])
    finally:
        f.close()
    
    f = open('train-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            label = struct.unpack('B',f.read(1))[0]
            train[i].append(label)
    finally:
        f.close()
        
    f = open('t10k-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        nrows = struct.unpack('>I',f.read(4))[0]
        ncols = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            data = struct.unpack('784B', f.read(nrows*ncols) )
            test.append([ data ])
    finally:
        f.close()
    
    f = open('t10k-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            label = struct.unpack('B',f.read(1))[0]
            test[i].append(label)
    finally:
        f.close()        
        
    return train,test,nrows,ncols