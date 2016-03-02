
from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    epoch = 10
    x,y = generate_linear_classification_data(n1=100,n2=100,mean1=[1.,1.],mean2=[4.,4.])
    logistic_scores = []
    knn_scores = []
    for i in range(epoch):
        print i
        _x,_y = resampling(x,y,ratio=0.8)
        cross_val_score = cross_validation(LogisticRegression(),_x,_y,10)
        logistic_scores.append(cross_val_score)
        cross_val_score = cross_validation(KNearestNeighbors(),_x,_y,10)
        knn_scores.append(cross_val_score)

    print 'logistic scores : mean = %f, std = %f' % (np.mean(logistic_scores),np.std(logistic_scores))
    print 'knn scores : mean = %f, std = %f' % (np.mean(knn_scores),np.std(knn_scores))

    _x,_y = shuffle(x,y)
    train_x,train_y,test_x,test_y = split_train_test(_x,_y)

    _x = train_x
    _y = train_y

    plt.plot(_x[np.where(_y.reshape(len(_y)) == 0)[0],0], _x[np.where(_y.reshape(len(_y)) == 0)[0],1],'ro')
    plt.plot(_x[np.where(_y.reshape(len(_y)) == 1)[0],0], _x[np.where(_y.reshape(len(_y)) == 1)[0],1],'bo')

    _x = test_x
    _y = test_y

    plt.plot(_x[np.where(_y.reshape(len(_y)) == 0)[0],0], _x[np.where(_y.reshape(len(_y)) == 0)[0],1],'r+')
    plt.plot(_x[np.where(_y.reshape(len(_y)) == 1)[0],0], _x[np.where(_y.reshape(len(_y)) == 1)[0],1],'b+')

    plt.show()