import numpy as np
import matplotlib.pyplot as plt

def generate(start=0.0,stop=1.0,step=0.1):
    x = np.arange(start,stop,step)
    f = lambda x: x*x
    y = np.array( [ f(x[i]) * np.random.normal(1,0.05) for i in range(len(x)) ] )
    return x,y

def cost(f,x,y):
    pass

def linear_regression(x,y):
    
    pass

if __name__ == '__main__':
    x,y = generate(0.0,10.0,0.1)
    plt.plot(x,y,'ro')
    plt.show()

'''
def generate(n=250,mu=1,sigma=0.01):
    prices = [ 1 ]
    for i in range(n):
        prices.append(prices[len(prices)-1] * 1.006)
    return prices

def retRate(p1,p2):
    return (p2-p1)/p1

if __name__ == '__main__':
    p1 = generate()
    p2 = generate(mu=1.0,sigma=0.000001)
    plt.plot(range(len(p1)),p1,'r-')
    plt.plot(range(len(p2)),p2,'r-')
    plt.show()

    r1 = []
    for i in range(1,len(prices)):
        rate.append(retRate(prices[i-1],prices[i]))
    
    plt.plot(range(len(p1)),prices,'r-')
    plt.plot(range(len(rate)),rate,'r-')
    plt.show()

def generate1(xs):
    d = [ np.sin(xs[i]) + np.random.normal(scale=0.2) for i in range(len(xs)) ]
    return d

if __name__ == '__main__':
    x = np.arange(0,12.0,0.1)
    y = generate(x)
    plt.plot(x,y,'o')
    plt.show()
'''