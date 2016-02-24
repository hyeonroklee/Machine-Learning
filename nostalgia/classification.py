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
        pass

    def train(self,x,y):
        t = np.append(x,np.ones((len(x),1)),axis=1)
        w = np.random.normal(size=(t.shape[1],1))
        learning_rate = 0.01
        delta_threshold = 0.01


        delta = np.zeros(t.shape[1])
        for i in range(len(t)):
            delta += (learning_rate * (y[i] - sigmoid(t[i],w))) * t[i]
        w += delta.reshape(w.shape)


    def predict(self,x):
        pass

    def score(self,x,y):
        pass


def logistic_regression(x,y):
    # full batcha
    t = np.append(x,np.matrix(np.ones(len(x))).T,axis=1)
    w = np.random.normal(size=(t.shape[1],1))
    alpha = 0.001 # learning rate
    delta = 0.01
    
    while True:
        delta1 = 0.
        delta2 = 0.
        delta3 = 0.
        for i in range(t.shape[0]):
            delta1 = delta1 + (alpha * ((y[i] - sigmoid(t[i],w)) * t[i,0]))[0,0]
            delta2 = delta2 + (alpha * ((y[i] - sigmoid(t[i],w)) * t[i,1]))[0,0]
            delta3 = delta3 + (alpha * ((y[i] - sigmoid(t[i],w)) * t[i,2]))[0,0]
        # check if there is a convergence to adjust the learning rate
        # print delta1,delta2,delta3
        if np.abs(delta1) < delta and np.abs(delta2) < delta and np.abs(delta3) < delta:
            break
        w[0] += delta1
        w[1] += delta2
        w[2] += delta3                
    return w

# decision tree

def decision_tree_calculate_entropy(x,y):
    # calculate entropy
    num = len(x)
    label_cnt = {}
    entropy = 0.
    for i in range(num):
        if not label_cnt.has_key(y[i]):
            label_cnt[y[i]] = 0
        label_cnt[y[i]] += 1
    for label in label_cnt.keys():
        prob = float(label_cnt[label]) / num
        entropy -= prob * np.log(prob)
    return entropy    

def decision_tree_split_data(x,y,idx,value):
    # splitting dataset
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    num = len(x)
    for i in range(num):
        if x[i][idx] == value:
            left_x.append( x[i][:idx] + x[i][idx+1:] )
            left_y.append( y[i])
        else:
            right_x.append( x[i][:idx] + x[i][idx+1:] )
            right_y.append( y[i])
    return left_x,left_y,right_x,right_y
    
def decision_tree_choose_feature_to_split(x,y):
    num_of_features = len(x[0])
    old_entropy = decision_tree_calculate_entropy(x,y)
    selected_idx = -1
    selected_value = -1
    for i in range(num_of_features):
        # extract values given a feature
        values = []
        for j in range(len(x)):
            values.append(x[j][i])
        values = set(values)
        
        for v in values:
            l_x,l_y,r_x,r_y = decision_tree_split_data(x,y,i,v)
            len_l_y = len(l_y)
            len_r_y = len(r_y)
            new_entropy = \
                decision_tree_calculate_entropy(l_x,l_y) * (float(len_l_y)/(len_l_y+len_r_y)) + \
                decision_tree_calculate_entropy(r_x,r_y) * (float(len_r_y)/(len_l_y+len_r_y))
            # print i,v,l_x, '############' , r_x, decision_tree_calculate_entropy(l_x,l_y),decision_tree_calculate_entropy(r_x,r_y)
            # print old_entropy,new_entropy
            if new_entropy < old_entropy:
                selected_idx = i
                selected_value = v
            
    return selected_idx,selected_value

def decision_tree(x,y,t):
    print x,y
    if len(x) < 2:
        return
    idx, value = decision_tree_choose_feature_to_split(x,y)
    if idx < 0 or value < 0:
        return
    l_x,l_y,r_x,r_y = decision_tree_split_data(x,y,idx,value)
    decision_tree(l_x,l_y,t)    
    decision_tree(r_x,r_y,t)
    


# class Kernel(object):
#     """Implements list of kernels from
#     http://en.wikipedia.org/wiki/Support_vector_machine
#     """
#     @staticmethod
#     def linear():
#         return lambda x, y: np.inner(x, y)
#
#     @staticmethod
#     def gaussian(sigma):
#         return lambda x, y: \
#             np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))
#
#     @staticmethod
#     def _polykernel(dimension, offset):
#         return lambda x, y: (offset + np.inner(x, y)) ** dimension
#
#     @classmethod
#     def inhomogenous_polynomial(cls, dimension):
#         return cls._polykernel(dimension=dimension, offset=1.0)
#
#     @classmethod
#     def homogenous_polynomial(cls, dimension):
#         return cls._polykernel(dimension=dimension, offset=0.0)
#
#     @staticmethod
#     def hyperbolic_tangent(kappa, c):
#         return lambda x, y: np.tanh(kappa * np.dot(x, y) + c)
#
#
# MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5
#
# class SVMTrainer(object):
#     def __init__(self, kernel, c):
#         self._kernel = kernel
#         self._c = c
#
#     def train(self, X, y):
#         """Given the training features X with labels y, returns a SVM
#         predictor representing the trained SVM.
#         """
#         lagrange_multipliers = self._compute_multipliers(X, y)
#         return self._construct_predictor(X, y, lagrange_multipliers)
#
#     def _gram_matrix(self, X):
#         n_samples, n_features = X.shape
#         K = np.zeros((n_samples, n_samples))
#         # TODO(tulloch) - vectorize
#         for i, x_i in enumerate(X):
#             for j, x_j in enumerate(X):
#                 K[i, j] = self._kernel(x_i, x_j)
#         return K
#
#     def _construct_predictor(self, X, y, lagrange_multipliers):
#         support_vector_indices = \
#             lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
#
#         support_multipliers = lagrange_multipliers[support_vector_indices]
#         support_vectors = X[support_vector_indices]
#         support_vector_labels = y[support_vector_indices]
#
#         # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
#         # bias = y_k - \sum z_i y_i  K(x_k, x_i)
#         # Thus we can just predict an example with bias of zero, and
#         # compute error.
#         bias = np.mean(
#             [y_k - SVMPredictor(
#                 kernel=self._kernel,
#                 bias=0.0,
#                 weights=support_multipliers,
#                 support_vectors=support_vectors,
#                 support_vector_labels=support_vector_labels).predict(x_k)
#              for (y_k, x_k) in zip(support_vector_labels, support_vectors)])
#
#         return SVMPredictor(
#             kernel=self._kernel,
#             bias=bias,
#             weights=support_multipliers,
#             support_vectors=support_vectors,
#             support_vector_labels=support_vector_labels)
#
#     def _compute_multipliers(self, X, y):
#         n_samples, n_features = X.shape
#
#         K = self._gram_matrix(X)
#         # Solves
#         # min 1/2 x^T P x + q^T x
#         # s.t.
#         #  Gx \coneleq h
#         #  Ax = b
#
#         P = cvxopt.matrix(np.outer(y, y) * K)
#         q = cvxopt.matrix(-1 * np.ones(n_samples))
#
#         # -a_i \leq 0
#         # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
#         G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
#         h_std = cvxopt.matrix(np.zeros(n_samples))
#
#         # a_i \leq c
#         G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
#         h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)
#
#         G = cvxopt.matrix(np.vstack((G_std, G_slack)))
#         h = cvxopt.matrix(np.vstack((h_std, h_slack)))
#
#         A = cvxopt.matrix(y, (1, n_samples))
#         b = cvxopt.matrix(0.0)
#
#         solution = cvxopt.solvers.qp(P, q, G, h, A, b)
#
#         # Lagrange multipliers
#         return np.ravel(solution['x'])
#
#
# class SVMPredictor(object):
#     def __init__(self,
#                  kernel,
#                  bias,
#                  weights,
#                  support_vectors,
#                  support_vector_labels):
#         self._kernel = kernel
#         self._bias = bias
#         self._weights = weights
#         self._support_vectors = support_vectors
#         self._support_vector_labels = support_vector_labels
#         assert len(support_vectors) == len(support_vector_labels)
#         assert len(weights) == len(support_vector_labels)
#         logging.info("Bias: %s", self._bias)
#         logging.info("Weights: %s", self._weights)
#         logging.info("Support vectors: %s", self._support_vectors)
#         logging.info("Support vector labels: %s", self._support_vector_labels)
#
#     def predict(self, x):
#         """
#         Computes the SVM prediction on the given features x.
#         """
#         result = self._bias
#         for z_i, x_i, y_i in zip(self._weights,
#                                  self._support_vectors,
#                                  self._support_vector_labels):
#             result += z_i * y_i * self._kernel(x_i, x)
#         return np.sign(result).item()
#
# def svm(x,y,t):
#     pass

