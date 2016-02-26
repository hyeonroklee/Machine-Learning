import numpy as np

def normalize(x):
    return (x - np.mean(x,axis=0)) / np.std(x,axis=0)

def pca_compress(x,k=1):
    data = np.matrix(x)
    evar,evec = np.linalg.eig(np.cov(data.T))
    evec = np.matrix(evec[:,evar.argsort()[::-1]])[:,np.arange(k)]
    res = data * evec
    return res,evec

def pca_decompress(x,evec):
    data = np.matrix(x)
    evec = np.matrix(evec)
    res = data * evec.T
    return res