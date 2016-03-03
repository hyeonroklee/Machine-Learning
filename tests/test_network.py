from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y = generate_linear_classification_data(n1=50,n2=50)
    _x,_y = shuffle(x,y)
    train_data = []
    test_data = []
    for i in range(len(_x)):
        train_data.append(
            [_x[i].reshape(2,1),
                [ [1],[0] ] if _y[i] == 0 else [ [0],[1] ] ])
    for i in range(len(_x)):
        test_data.append(
            [_x[i].reshape(2,1), _y[i] ])

    layers = [2,5,2]
    n = Network(layers)
    n.SGD(train_data,200,len(train_data),0.001)
    print n.evaluate(test_data)
