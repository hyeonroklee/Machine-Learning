import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import samples_generator
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC

def test_linear_regression():
    x,y = samples_generator.make_regression(n_features=1,noise=10.)
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    predictor = LinearRegression()
    predictor.fit(x_train,y_train)
    print predictor.score(x_test,y_test),predictor.intercept_,predictor.coef_

    cv = cross_val_score(LinearRegression(),x,y,cv=10)
    print np.mean(cv),cv

def test_logistic_regression():
    x,y = samples_generator.make_classification(n_features=2,n_redundant=0)
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    predictor = LogisticRegression()
    predictor.fit(x_train,y_train)
    print predictor.score(x_test,y_test),predictor.intercept_,predictor.coef_

def test_svm():
    x,y = samples_generator.make_classification(n_features=2,n_redundant=0)
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    predictor = SVC()
    predictor.fit(x_train,y_train)
    print predictor.score(x_test,y_test)

if __name__ == '__main__':
    # test_linear_regression()
    # test_logistic_regression()
    test_svm()