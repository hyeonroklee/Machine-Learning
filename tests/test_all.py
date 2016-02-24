
from nostalgia import *

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    a = generate_random_walk()
    train_x,train_y,test_x,test_y = read_mnist_digit()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # classifier = KNearestNeighbors()
    # classifier.train(train_x[:10000],train_y[:10000])
    # print classifier.score(test_x[:100],test_y[:100])
    #
    # train_y = np.array(train_y).reshape(len(train_x))
    # test_y = np.array(test_y).reshape(len(test_x))
    #
    # classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier.fit(train_x[:10000],train_y[:10000])
    #
    # print classifier.score(test_x[:100],test_y[:100])

    classifier = LogisticRegression()
    classifier.train(train_x,train_y)
