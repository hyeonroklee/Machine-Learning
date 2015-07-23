# -*- coding: utf-8 -*-

import numpy as np

def split_train_test(x,y=None,ratio=0.75):
    split_size = len(x)*ratio
    if y is not None:
        if len(x) != len(y):
            raise Exception('dimension unmatched')
        return x[:split_size],x[split_size:],y[:split_size],y[split_size:]
    else:
        return x[:split_size],x[split_size:]

def bootstrapping(x,y=None,w=None,ratio=1.):
    sampling_size = int(len(x)*ratio)
    x_sampling = []
    y_sampling = []

    if w is None:
        w = np.cumsum(np.repeat(1./float(len(x)),len(x)))
    c = np.cumsum(w)

    for i in range(sampling_size):
        idx = np.where(c>=np.random.uniform())[0][0]
        x_sampling.append(x[idx])
        y_sampling.append(y[idx])

    return x_sampling,y_sampling


def cross_validation(d,k):
    if len(d) < k or len(d) % k != 0:
        raise Exception('dimension unmatched')
    data = np.array(d)
    for i in range(len(d) / k):
        print data[i*k:i*k+k],data[:i*k],data[i*k+k:len(d)]