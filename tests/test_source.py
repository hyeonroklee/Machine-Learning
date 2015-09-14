from nostalgia.data import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    y = generate_random_walk(100,[0,0],[[0.015,0.],[0.,0.015]])
    x = np.arange(0,len(y))
    for i in range(y.shape[1]):
        plt.plot(x,y[:,i])
    plt.show()