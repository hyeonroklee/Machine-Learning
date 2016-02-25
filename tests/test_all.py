
from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    x,y = generate_linear_regression_data()
    regression = LinearRegression(ne=False)
    w = regression.train(x,y)

    t_x = np.linspace(-10,10,50).reshape(50,1)
    t_y = regression.predict(t_x)

    plt.plot(x,y,'ro')
    plt.plot(t_x,t_y)
    plt.show()


    # x,y = generate_linear_classification_data(n1=50,n2=50)
    #
    # classifier = LogisticRegression()
    # w = classifier.train(x,y)

    # t = np.array([ [1,1],[3,3] ])
    # print classifier.predict(t)
    # print classifier.score(x,y)

    # plt.plot( x[np.where(y.reshape(len(y)) == 0)[0],0], x[np.where(y.reshape(len(y)) == 0)[0],1],'ro')
    # plt.plot( x[np.where(y.reshape(len(y)) == 1)[0],0], x[np.where(y.reshape(len(y)) == 1)[0],1],'bo')
    # plt.plot(np.linspace(0,5),(-w[2]-w[0]*np.linspace(0,5))/w[1])
    # plt.show()


    # train_x,train_y,test_x,test_y = read_mnist_digit()
    #
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    # test_x = np.array(test_x)
    # test_y = np.array(test_y)
    #
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
