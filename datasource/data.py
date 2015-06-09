
import pylab, scipy.io
import struct
import numpy as np
import matplotlib.pyplot as plt

def createMultiNormalDataSet(n=100):
    mus = np.array( [0.0, 0.0] )
    cov = np.array( [ [1.0, 0.5], [0.5, 1.0] ] )
    return np.random.multivariate_normal(mus,cov,n)

def readDigitTrain():
    result = []
    
    f = open('datasource/train-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))
        numOfItems = struct.unpack('>I',f.read(4))
        nrows = struct.unpack('>I',f.read(4))
        ncols = struct.unpack('>I',f.read(4))
        print('numOfItems = ',numOfItems, ' nrows = ',nrows, ' ncols = ',ncols)
        for i in range(numOfItems[0]):
            data = struct.unpack('784B', f.read(nrows[0]*ncols[0]) )
            result.append([ data ])
    finally:
        f.close()
    
    f = open('datasource/train-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))
        numOfItems = struct.unpack('>I',f.read(4))
        print('numOfItems = ',numOfItems)
        for i in range(numOfItems[0]):
            label = struct.unpack('B',f.read(1))
            result[i].append(label)
    finally:
        f.close()
        
    return result

def readDigitTest():
    result = []
    f = open('datasource/t10k-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))
        numOfItems = struct.unpack('>I',f.read(4))
        nrows = struct.unpack('>I',f.read(4))
        ncols = struct.unpack('>I',f.read(4))
        print('numOfItems = ',numOfItems, ' nrows = ',nrows, ' ncols = ',ncols)
        for i in range(numOfItems[0]):
            data = struct.unpack('784B', f.read(nrows[0]*ncols[0]) )
            result.append([ data ])
    finally:
        f.close()
    
    f = open('datasource/t10k-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))
        numOfItems = struct.unpack('>I',f.read(4))
        print('numOfItems = ',numOfItems)
        for i in range(numOfItems[0]):
            label = struct.unpack('B',f.read(1))
            result[i].append(label)
    finally:
        f.close()
    
    """
    plt.imshow(np.reshape(np.array(result[0][0]),(nrows[0],ncols[0])),cmap = pylab.gray())
    plt.show()
    print(result[0][1])
    """
    
    return result
    
def readImageMat():
    images = scipy.io.loadmat('datasource/IMAGES.mat')

if __name__ == '__main__':
    readDigitTest()