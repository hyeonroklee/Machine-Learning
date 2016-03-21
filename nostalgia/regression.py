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

    def train(self,x,y,alpha=0.001,batch_size=10,epochs=1):
        if self._ne:
            t = np.matrix(np.append(x,np.ones((len(x),1)),axis=1))
            y = np.matrix(y)
            self._w = np.linalg.inv(t.T * t) * (t.T * y)
        else:
            t = np.matrix(np.append(x,np.ones((len(x),1)),axis=1))
            y = np.matrix(y)
            w = np.matrix(np.random.normal(size=(t.shape[1],1)))
            prev_delta = None
            while True:
                delta = np.matrix(np.zeros(t.shape[1]))
                for i in np.random.randint(0,len(t),batch_size):
                    delta += (alpha * ((y[i] - t[i] * w) * t[i]))
                if prev_delta is None:
                    prev_delta = delta
                else:
                    if np.sum(np.abs(prev_delta)) < np.sum(np.abs(delta)):
                        alpha /= 2.0
                        continue
                if np.all(np.abs(delta) < 0.001) or np.any(np.isnan(delta)):
                    break
                w += delta.reshape(w.shape)

            self._w = w
        return self._w

    def predict(self,x):
        t = np.matrix(np.append(x,np.ones((len(x),1)),axis=1))
        return t * self._w

    def score(self,x,y):
        pass
