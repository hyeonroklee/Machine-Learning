# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np

def sigmoid(x,w):
    return 1. / (1.+np.exp(-np.dot(x,w)))
    
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def logistic_regression(x,y):
    # full batch
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

def knn(x,y,t,k=10):
    d = np.sqrt(np.sum( (np.array(x) - np.array(t)) * (np.array(x) - np.array(t)),axis=1) )
    idx = d.argsort()
    votes = {}
    for i in range(k):
        v = y[idx[i],0]
        votes[v] = votes.get(v,0) + 1
    print votes


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

def decision_tree(x,y):
    print x,y
    if len(x) < 2:
        return
    idx, value = decision_tree_choose_feature_to_split(x,y)
    if idx < 0 or value < 0:
        return
    l_x,l_y,r_x,r_y = decision_tree_split_data(x,y,idx,value)
    decision_tree(l_x,l_y)    
    decision_tree(r_x,r_y)
    
def svm():
    pass

if __name__ == '__main__':
    x = [ [1,0,1] , [0,1,0], [1,0,0 ] ]
    y = [ 1,0,0 ]
    decision_tree(x,y)
    
