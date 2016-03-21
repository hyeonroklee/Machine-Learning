# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod
import numpy as np

def sigmoid(x,w):
    return 1. / (1.+np.exp(-np.dot(x,w)))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Classifier:
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


class KNearestNeighbors(Classifier):
    def __init__(self,n_neighbors=10):
        super(KNearestNeighbors,self).__init__()
        self._n_neighbors = n_neighbors

    def train(self,x,y):
        self._train_x = x
        self._train_y = y

    def predict(self,x):
        ps = []
        for t in x:
            d = np.sqrt(np.sum((np.array(self._train_x) - np.array(t)) * (np.array(self._train_x) - np.array(t)),axis=1))
            idx = d.argsort()
            votes = {}
            for  i in range(self._n_neighbors):
                v = self._train_y[idx[i],0]
                votes[v] = votes.get(v,0) + 1
            p = max(votes,key=lambda k: votes[k])
            ps.append(p)
        return np.array(ps).reshape(len(ps),len(ps[0]) if isinstance(ps[0],list) else 1)

    def score(self,x,y):
        predict_y = self.predict(x)
        return 1 - float(np.count_nonzero(predict_y - y)) / len(x)


class LogisticRegression(Classifier):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self._w = None

    def train(self,x,y,alpha=0.01,batch_size=10,epochs=1):
        t = np.matrix(np.append(x,np.ones((len(x),1)),axis=1))
        y = np.matrix(y)
        w = np.matrix(np.random.normal(size=(t.shape[1],1)))
        prev_delta = None
        while True:
            delta = np.matrix(np.zeros(t.shape[1]))
            for i in range(len(t)):
                delta += (alpha * ((y[i] - sigmoid(t[i],w)) * t[i]))
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
        return w

    def predict(self,x):
        ps = []
        for a in x:
            ps.append(1 if sigmoid(np.append(a,1),self._w) >= 0.5 else 0)
        return np.array(ps).reshape(len(ps),1)

    def score(self,x,y):
        predict_y = self.predict(x)
        return 1 - float(np.count_nonzero(predict_y - y)) / len(x)


class DecisionTree(Classifier):
    def __init__(self):
        super(DecisionTree,self).__init__()

    def train(self,x,y):
        if len(x) < 2:
            return
        idx, value = self.choose_feature_to_split(x,y)
        if idx < 0 or value < 0:
            return
        l_x,l_y,r_x,r_y = self.split_data(x,y,idx,value)
        # decision_tree(l_x,l_y,t)
        # decision_tree(r_x,r_y,t)

    def predict(self,x):
        pass

    def score(self,x,y):
        pass

    def calculate_entropy(self,x,y):
        label_cnt = {}
        entropy = 0
        for i in range(len(x)):
            label_cnt[y[i][0]] = label_cnt.get(y[i][0],0) + 1
        for label in label_cnt:
            prob = float(label_cnt[label]) / len(x)
            entropy -= prob * np.log(prob)
        return entropy

    def split_data(self,x,y,index,value):
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for i in range(len(x)):
            if x[i][index] == value:
                left_x.append( x[i][:index] + x[i][index+1:] )
                left_y.append( y[i])
            else:
                right_x.append( x[i][:index] + x[i][index+1:] )
                right_y.append( y[i])
        return left_x,left_y,right_x,right_y

    def choose_feature_to_split(self,x,y):
        num_of_features = len(x[0])
        old_entropy = self.calculate_entropy(x,y)
        selected_idx = -1
        selected_value = -1
        for i in range(num_of_features):
            # extract values given a feature
            values = []
            for j in range(len(x)):
                values.append(x[j][i])
            values = set(values)

            for v in values:
                l_x,l_y,r_x,r_y = self.split_data(x,y,i,v)
                len_l_y = len(l_y)
                len_r_y = len(r_y)
                new_entropy = \
                    self.calculate_entropy(l_x,l_y) * (float(len_l_y)/(len_l_y+len_r_y)) + \
                    self.calculate_entropy(r_x,r_y) * (float(len_r_y)/(len_l_y+len_r_y))
                # print i,v,l_x, '############' , r_x, decision_tree_calculate_entropy(l_x,l_y),decision_tree_calculate_entropy(r_x,r_y)
                # print old_entropy,new_entropy
                if new_entropy < old_entropy:
                    selected_idx = i
                    selected_value = v

        return selected_idx,selected_value