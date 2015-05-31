# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:55:16 2015

@author: hyeonrok lee
"""

import numpy as np

def pca_compress(data,k=1):
    data = np.matrix(data)
    evar,evec = np.linalg.eig(np.cov(data.T))
    evec = np.matrix(evec[:,evar.argsort()])[:,np.arange(k)]
    res = data * evec
    return res,evec
 
def pca_decompress(data,evec):
    data = np.matrix(data)
    evec = np.matrix(evec)
    res = data * evec.T
    return res