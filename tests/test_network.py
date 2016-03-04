from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y = generate_linear_classification_data(n1=50,n2=50)
    _x,_y = shuffle(x,y)
    train_x,train_y,test_x,test_y = split_train_test(_x,_y)

    train_data = []
    test_data = []
    for i in range(len(train_x)):
        train_data.append(
            [train_x[i].reshape(2,1),
                np.array([[1],[0]] if train_y[i] == 0 else [[0],[1]]) ])

    layers = [2,2]
    n = Network(layers)
    n.SGD(train_data,500,len(train_data)/2,0.01,monitor_training_cost=True,monitor_training_accuracy=True)

    for i in range(len(test_x)):
        test_data.append(
            [test_x[i].reshape(2,1), np.array(test_y[i]) ])

    print n.accuracy(test_data)

    # plt.plot(_x[np.where(_y.reshape(len(_y)) == 0)[0],0], _x[np.where(_y.reshape(len(_y)) == 0)[0],1],'ro')
    # plt.plot(_x[np.where(_y.reshape(len(_y)) == 1)[0],0], _x[np.where(_y.reshape(len(_y)) == 1)[0],1],'bo')
    # plt.show()