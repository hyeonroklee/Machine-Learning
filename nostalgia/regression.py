# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod

import numpy as np

class Regression:
    __metaclass__  = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def train(self,x,y):
        pass

    @abstractmethod
    def predict(self,x):
        pass

    @abstractmethod
    def score(self,x,y):
        pass


class LinearRegression(Regression):
    def __init__(self,ne=True):
        super(LinearRegression,self).__init__()
        self._w = None
        self._ne = ne

    def train(self,x,y):
        if self._ne:
            t = np.matrix(np.append(x,np.ones((len(x),1)),axis=1))
            y = np.matrix(y)
            self._w = np.linalg.inv(t.T * t) * (t.T * y)
        else:
            t = np.append(x,np.ones((len(x),1)),axis=1)
            w = np.random.normal(size=(t.shape[1],1))
            learning_rate = 0.001
            while True:
                delta = np.zeros(t.shape[1])
                for i in range(len(t)):
                    delta += (learning_rate * (y[i] - np.dot(t[i],w))) * t[i]
                if np.all(np.abs(delta) < 0.001):
                    break
                w += delta.reshape(w.shape)
            self._w = w
            return w

    def predict(self,x):
        t = np.matrix(np.append(x,np.ones((len(x),1)),axis=1))
        return t * self._w

    def score(self,x,y):
        pass
