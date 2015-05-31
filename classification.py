# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:54:41 2015

@author: hyeonrok lee
"""

def logistic_reg():
    pass

def knn_classify(dataset,x,k):
    dataSetSize = len(dataSet)
    features = np.array( [ [a[0],a[1]] for a in dataSet ] )
    distMat = np.tile(x,(dataSetSize,1)) - features
    distMat = (np.sum(distMat**2,axis=1))**0.5
    distIdx = distMat.argsort()
    voteCount = {}
    for i in range(k):
        vote = dataSet[distIdx[i]][2]
        voteCount[vote] = voteCount.get(vote,0) + 1
    sortedVoteCount = sorted(voteCount.iteritems(),key=op.itemgetter(1), reverse=True)
    return sortedVoteCount[0][0]