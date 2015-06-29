# -*- coding: utf-8 -*-
"""

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

    