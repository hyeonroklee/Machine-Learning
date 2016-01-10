import numpy as np

def generate_linear_regression_data(n=20,low_bound=-10,high_bound=10,noize=1,f = lambda x : x + 2):
    x = np.array([ np.random.uniform(low_bound,high_bound,n) ])
    y = f(x) + np.random.normal(0,noize,n)

    return x.T,y.T

def generate_linear_classification_data(n1=10,n2=10):
    mean1 = [1.,1.]
    cov1 = [ [ 1., 0.5 ] , [ 0.5, 1. ] ]
    d1 = np.random.multivariate_normal(mean1,cov1,n1)
    mean2 = [3.,3.]
    cov2 = [ [ 1., 0.5 ] , [ 0.5, 1. ] ]
    d2 = np.random.multivariate_normal(mean2,cov2,n2)
    x = np.append(d1.T,d2.T,axis=1)
    y = np.append([np.zeros(n1)],[np.ones(n2)],axis=1)
    return x.T,y.T