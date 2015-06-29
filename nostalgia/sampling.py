# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def cross_validation(d,k):
    if len(d) < k or len(d) % k != 0:
        raise Exception('dimention unmatched')
    data = np.array(d)
    for i in range(len(d) / k):
        print data[i*k:i*k+k],data[:i*k],data[i*k+k:len(d)]

def bootstrapping(d):
    data = np.array(d)
    return data[np.random.randint(0,len(d),len(d))]

if __name__ == '__main__':
    d = [ [1,2], [3,4], [5,6] ]
    cross_validation(d,1)
'''
    try:
        cross_validation(d,1)
    except Exception as e:
        print str(e)
'''