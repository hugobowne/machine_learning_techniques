__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from wine_data import WineData
from wine_preprocesser import WinePreprocesser
from wine_explore import plot2d, pairs

from time import time

from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

# Load data and preprocess (everything you don't put in the pipeline)
data = WineData('../winequality-red.csv', '../winequality-white.csv')

print('Preprocesing.')
t0 = time() 
wp = WinePreprocesser(data)
wp.add_divided_features(replace_inf_with_absmax=True)
wp.polynomial_expansion(rank=2)
wp.remove_low_variance_features(variance_threshold=0)
X_red, y_red = wp.get_red()
X_white, y_white = wp.get_white()
print('Preprocesing. Done in %fs' % (time()-t0) )
###############################
# Explore data 
# 1. Plot in 2d, color code classes: 
#	-> no simple low dimension linear separation
# 2. Plot paris 
#	-> correlation: transform data or use regularized methods
#	-> non-normal distributed featues: Box-Cox transform
###############################
do_plot = True 
if (do_plot): 
	plot2d(X_red, y_red, embedding='gallery', title='Red wine').show()#.savefig('../data/red_whine_2d_gallery.png')
	plot2d(X_white, y_white, embedding='gallery', title='White wine').show()#.savefig('../data/white_whine_2d_gallery.png')
	pairs(X_red, y_red, 'Red wine')
	pairs(X_white, y_white, 'White wine')

# ###############################
# # Classification 
# # Prepare data 
# ###############################
# #X = X_white
# #y = y_white
# X = X_red
# y = y_red
# X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, random_state=23, test_size=0.2)

# ###############################
# # What is a base level classification? 
# # Not sure what to use, i) nearest neighbor classification, ii) logisitic regression, iii) linear SVM iv) Naive Bayes ? 
# # Best score: Red wine: NB: 0.55 (Gaussian), NN: 0.63 (n_neighbors=1), SVC: 0.57 (C=10), Logisitic regression 0.59 (C=1)
# # Best score: White wine: NB: 0.43 (Gaussian), NN: 0.63 (n_neighbors=1), SVC: 0.53 (C=10), Logisitic regression 0.54 (C=0.1)
# ###############################
# from sklearn.metrics import accuracy_score
# highest_class = pd.Series(y_red).value_counts().order().index[0]
# #print( accuracy_score((np.repeat(highest_class, len(y_red), y_red))) )

# def train_plot_single_param_grid(steps, param_grid, name): 
# 	"""
# 	Train model and plot the learning curve. 
# 	This works only if there is one parameter to optimize over. 
# 	"""
# 	clf = Pipeline(steps)
# 	grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=2, scoring='accuracy').fit(X_train, y_train)
# 	print( ('Best score %s of estimator %s') % (grid_search.best_score_, grid_search.best_estimator_))
# 	plt.figure(1)
# 	plt.title("Learning curve " + name)
# 	plt.plot([c.mean_validation_score for c in grid_search.grid_scores_], label="validation error")
# 	plt.show()
# 	return grid_search.best_estimator_

# def evaluate(classifier, valid_X, valid_y):
# 	"""
# 	Tests model on validation data set

# 	Parameters
# 	----------

# 	classifier : classifier
# 		Learned model 

# 	valid_X : array 
# 		Text to classify 

# 	valid_y : array 
# 		Label of text
# 	"""
# 	predicted = classifier.predict(valid_X)
# 	print(metrics.classification_report(valid_y, predicted))

# do_base_rate = False
# if (do_base_rate): 
# 	steps = [('scale', StandardScaler()), ('nb', GaussianNB())]
# 	scores = cross_validation.cross_val_score(Pipeline(steps), X_train, y_train, cv=5)
# 	print(('Best score %s') % (scores.mean()))

# 	steps = [('scale', StandardScaler()), ('nn', KNeighborsClassifier())]
# 	param_grid = {'nn__n_neighbors': [1, 2, 4, 8, 32, 64]}
# 	clf = train_plot_single_param_grid(steps, param_grid, 'Nearest Neighbor')
# 	evaluate(clf, X_holdout, y_holdout)

# 	steps = [('scale', StandardScaler()), ('svc', LinearSVC())]
# 	param_grid = {'svc__C': 10. ** np.arange(-3, 4)}
# 	clf = train_plot_single_param_grid(steps, param_grid, 'Linear SVC')
# 	evaluate(clf, X_holdout, y_holdout)

# 	steps = [('scale', StandardScaler()), ('logistic', LogisticRegression(multi_class='multinomial', solver='lbfgs'))]
# 	param_grid = {'logistic__C': 10. ** np.arange(-3, 4)}
# 	clf = train_plot_single_param_grid(steps, param_grid, 'Logistic')
# 	evaluate(clf, X_holdout, y_holdout)

# 	steps = [('scale', StandardScaler()), ('trans', PCA()), ('nn', KNeighborsClassifier())]
# 	param_grid = {'trans__n_components': np.arange(2,X_train.shape[1]+1, 2), 'nn__n_neighbors': [1, 2, 4, 8, 32, 64]}
# 	clf = train_plot_single_param_grid(steps, param_grid, 'Nearest Neighbor')
# 	evaluate(clf, X_holdout, y_holdout)

# steps = [('scale', StandardScaler()), ('trans', PCA()), ('clf', RandomForestClassifier())]
# param_grid = {'trans__n_components': np.arange(2,X_train.shape[1]+1, 2), 'clf__n_estimators': [160, 320, 640]}
# clf = train_plot_single_param_grid(steps, param_grid, 'RF')
# evaluate(clf, X_holdout, y_holdout)

