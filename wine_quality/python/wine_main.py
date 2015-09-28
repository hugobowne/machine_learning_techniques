__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from wine_data import WineData
from wine_explore import plot2d, pairs

from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

data = WineData('../winequality-red.csv', '../winequality-white.csv')

X_red, y_red = data.load_red()
X_white, y_white = data.load_white()

###############################
# Explore data 
# 1. Plot in 2d, color code classes: 
#	-> no simple low dimension linear separation
# 2. Plot paris 
#	-> correlation: transform data or use regularized methods
#	-> non-normal distributed featues: Box-Cox transform
###############################
do_plot = False 
if (do_plot): 
	plot2d(X_red, y_red, embedding='gallery', title='Red wine').savefig('../data/red_whine_2d_gallery.png')
	plot2d(X_white, y_white, embedding='gallery', title='White wine').savefig('../data/white_whine_2d_gallery.png')
	pairs(X_red, y_red, 'Red wine')
	pairs(X_white, y_white, 'White wine')

###############################
# Classification 
# Prepare data 
###############################
X = X_red
y = y_red
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, random_state=23, test_size=0.1)

###############################
# What is a base level classification? 
#
# Would be cool, to implement a simple zero  
# Not sure what to use, i) nearest neighbor classification, ii) logisitic regression, iii) linear SVM iv) Naive Bayes ? 
# Best score: NN: 0.63 (n_neighbors=1), SVC: 0.57 (C=10), Logisitic regression 0.59 (C=1)
###############################
do_base_rate = True
if (do_base_rate): 
	def train_plot_single_param_grid(steps, param_grid, name): 
		clf = Pipeline(steps)
		grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1).fit(X_train, y_train)
		print( ('Best score %s of estimator %s') % (grid_search.best_score_, grid_search.best_estimator_))
		plt.figure(1)
		plt.title("Learning curve " + name)
		plt.plot([c.mean_validation_score for c in grid_search.grid_scores_], label="validation error")
		plt.show()

	steps = [('scale', StandardScaler()), ('nb', GaussianNB())]
	scores = cross_validation.cross_val_score(Pipeline(steps), X_train, y_train, cv=5)
	print(('Best score %s') % (scores.mean()))

	steps = [('scale', StandardScaler()), ('nn', KNeighborsClassifier())]
	param_grid = {'nn__n_neighbors': [1, 2, 4, 8, 32, 64, 128, 256, 512]}
	train_plot_single_param_grid(steps, param_grid, 'Nearest Neighbor')

	steps = [('scale', StandardScaler()), ('svc', LinearSVC())]
	param_grid = {'svc__C': 10. ** np.arange(-3, 4)}
	train_plot_single_param_grid(steps, param_grid, 'Linear SVC')

	steps = [('scale', StandardScaler()), ('logistic', LogisticRegression(multi_class='multinomial', solver='lbfgs'))]
	param_grid = {'logistic__C': 10. ** np.arange(-3, 4)}
	train_plot_single_param_grid(steps, param_grid, 'Logistic')

###############################
# What is a base level classification? 
# Not sure what to use, i) nearest neighbor classification, ii) logisitic regression, iii) linear SVM? 
# Best score: NN: 0.63 (n_neighbors=1), SVC: 0.57 (C=10), Logisitic regression 0.59 (C=1)
###############################

