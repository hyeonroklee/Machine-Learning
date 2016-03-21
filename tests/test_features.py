from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_x,train_y = generate_linear_regression_data(n=100,f = lambda x : x**3 + 10*(x**2) + 2*x + 2)
    regression = LinearRegression(ne=False)
    poly_train_x = polynomial(train_x)
    w = regression.train(poly_train_x,train_y,alpha=0.000001,batch_size=len(poly_train_x))

    test_x = np.linspace(-10,10,100).reshape(100,1)
    poly_test_x = polynomial(test_x)
    test_y = regression.predict(poly_test_x)

    plt.plot(train_x,train_y,'ro')
    plt.plot(test_x,test_y)
    plt.show()

    # x,y = generate_linear_classification_data(n1=50,n2=50)
    # compressed_x,vec = pca_compress(x)
    # res = pca_decompress(compressed_x,vec)
    #
    # plt.plot( x[np.where(y.reshape(len(y)) == 0)[0],0], x[np.where(y.reshape(len(y)) == 0)[0],1],'ro')
    # plt.plot( x[np.where(y.reshape(len(y)) == 1)[0],0], x[np.where(y.reshape(len(y)) == 1)[0],1],'bo')
    #
    # plt.plot( res[np.where(y.reshape(len(y)) == 0)[0],0], res[np.where(y.reshape(len(y)) == 0)[0],1],'ro')
    # plt.plot( res[np.where(y.reshape(len(y)) == 1)[0],0], res[np.where(y.reshape(len(y)) == 1)[0],1],'bo')
    #
    # plt.show()
