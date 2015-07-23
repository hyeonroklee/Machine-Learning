import numpy as np

import nostalgia.sampling as sa
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    x = np.array([[1,2], [3,4], [5,6], [7,8]])
    y = np.array([1,2,3,4])

    # x_train,x_test = sa.split_train_test(x)
    x,y = sa.bootstrapping(x,y)

    print x,y
    # print y_train,y_test
