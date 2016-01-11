
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nostalgia import *

if __name__ == '__main__':
    reg_x,reg_y = simulator.generate_linear_regression_data(n=3,low_bound=-3,high_bound=3,noize=0.1,f=lambda x: np.sin(x))
    print reg_x
    print reg_y

    x,y = sampling.shuffle(reg_x,reg_y)
    print x
    print y

    # norm_reg_x = features.normalize(reg_x)
    # tran_x, tran_y, test_x, test_y = sampling.split_train_test(norm_reg_x,reg_y)
    # w = regression.linear_regression_by_normal_equation(tran_x,tran_y)
    #
    # x = np.linspace(np.min(tran_x.T[0]),np.max(tran_x.T[0]),100)
    # y = (np.matrix([x,np.ones(len(x))]).T) * w
    # plt.plot(x,y.T.tolist()[0])
    # plt.plot(tran_x.T[0],tran_y.T[0],'ro')
    #
    # plt.show()
