import numpy as np

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def pca_compress(x,k=1):
    data = np.matrix(x)
    evar,evec = np.linalg.eig(np.cov(data.T))
    evec = np.matrix(evec[:,evar.argsort()])[:,np.arange(k)]
    res = data * evec
    return res,evec

def pca_decompress(x,evec):
    data = np.matrix(x)
    evec = np.matrix(evec)
    res = data * evec.T
    return res