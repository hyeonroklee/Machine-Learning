from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y = generate_linear_classification_data(n1=50,n2=50)
    print polynomial(x)

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
