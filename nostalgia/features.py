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

def polynomial(x,degree=2):
    _x = np.append(x,np.ones((len(x),1)),axis=1)
    poly_features = []
    for r in _x:
        converted_feature = r
        for i in range(degree-1):
            converted_feature = np.polymul(r,converted_feature)
        poly_features.append(converted_feature)
    return np.array(poly_features)[:,:-1]

class Digitizer(object):
    def __init__(self,x,n_bins=10):
        self._n_bins = n_bins
        self._min_max = []
        for i in range(x.shape[1]):
            c = x[:,i]
            min_value = np.min(c)
            max_value = np.max(c)
            self._min_max.append(np.array([min_value,max_value]))
        self._min_max = np.array(self._min_max)

    def transform(self,x):
        _result = []
        for i in range(x.shape[1]):
            c = x[:,i]
            min_value = self._min_max[i][0]
            max_value = self._min_max[i][1]
            _result.append(np.digitize(c,np.linspace(min_value,max_value,self._n_bins)))
        _result = np.array(_result).T
        return _result