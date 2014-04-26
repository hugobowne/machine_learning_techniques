# -*- coding: utf-8 -*-
"""
================================================================================
Digit recognition: Support vector machine parameter estimation using grid search 
================================================================================

Here I implemented a cross-validation algorithm. I used scikit learn's
`sklearn.grid_search.GridSearchCV` to train each classifier on half the
labeled data and used the other half as the cross-validation set to test
the performance of the classifier.

The classifiers I tested were all support vector machines (SVMs): Gaussian,
linear, and polynomial (degrees 2,3 and 4) over a range of parameters.

I tested these classifiers for precision, that is, the positive predictive
value or the proportion of those tested that are predicted correctly.
"""

from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm, metrics, preprocessing
import csv
import time

print(__doc__)

# Loading the Digits dataset


###
from numpy import genfromtxt


my_data = genfromtxt('train.csv', delimiter=',')


x_train = my_data[1:,1:]
x_train = preprocessing.scale(x_train)
t_train = my_data[1:,0]



start_time = time.time()
# Split the dataset in two equal parts
x_train, x_cv, t_train, t_cv = train_test_split(
    x_train, t_train, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10]},
                    {'kernel': ['linear'], 'C': [1, 10]},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [2]},
                    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [3]},
                    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [4]}]


scores = ['precision'] # you can alter this by adding, for example, `recall'

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters)
    clf.fit(x_train, t_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %s"
              % (mean_score, scores.std() / 2, params))
    print()


print(time.time()- start_time)


# <codecell>


# <codecell>


# <codecell>


