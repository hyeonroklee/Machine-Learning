import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import samples_generator
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

def test_linear_regression():
    x,y = samples_generator.make_regression(n_features=1,noise=10.)
    print x
    print y

# spam filter
# def test_logistic_regression():
#     df = pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)
#     X_train_raw,X_test_raw,y_train,y_test = train_test_split(df[1],df[0])
#     vectorize = TfidfVectorizer()
#     vectorize.fit(df[1])
#     X_train = vectorize.transform(X_train_raw)
#     X_test = vectorize.transform(X_test_raw)
#     classifier = LogisticRegression()
#     classifier.fit(X_train,y_train)
#     predictions = classifier.predict(X_test)
#
#     print y_test
#     print predictions
#
#     # print vectorize.vocabulary_
#     # print vectorize.fit_transform(X_train_raw).todense()
#     # print vectorize.vocabulary_
#
# def test_decision_tree():
#     pass
#
# def test_perceptron():
#     pass

if __name__ == '__main__':
    test_linear_regression()