# -*- coding: utf-8 -*-

from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':    

    x,y = generate_linear_classification_data(n1=50,n2=50)

    classifier = LogisticRegression()
    w = classifier.train(x,y)

    t = np.array([ [1,1],[3,3] ])
    print classifier.predict(t)
    print classifier.score(x,y)

    plt.plot( x[np.where(y.reshape(len(y)) == 0)[0],0], x[np.where(y.reshape(len(y)) == 0)[0],1],'ro')
    plt.plot( x[np.where(y.reshape(len(y)) == 1)[0],0], x[np.where(y.reshape(len(y)) == 1)[0],1],'bo')
    plt.plot(np.linspace(0,5),(-w[2,0]-w[0,0]*np.linspace(0,5))/w[1,0])
    plt.show()