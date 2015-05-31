# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:10:40 2015

@author: hyeonrok lee
"""

import numpy as np
import matplotlib.pyplot as plt

def regression_model(x,p):
    x = np.array(x)
    p = np.array(p)
    y = x.dot(p) + np.random.normal(size=(x.shape[0],p.shape[1]))
    return y


def classification_model():
    pass

if __name__ == '__main__':
    