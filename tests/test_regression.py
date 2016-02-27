
from nostalgia import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y = generate_linear_regression_data()
    regression = LinearRegression(ne=False)
    w = regression.train(x,y)

    t_x = np.linspace(-10,10,50).reshape(50,1)
    t_y = regression.predict(t_x)

    plt.plot(x,y,'ro')
    plt.plot(t_x,t_y)
    plt.show()
