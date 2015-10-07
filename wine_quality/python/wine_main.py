__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from wine_data import WineData
from wine_preprocesser import WinePreprocesser
from wine_explore import plot2d, pairs
from wine_classifier import WineClassifier

from time import time
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

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
do_plot = False 
if (do_plot): 
	plot2d(X_red, y_red, embedding='gallery', title='Red wine').show()#.savefig('../data/red_whine_2d_gallery.png')
	plot2d(X_white, y_white, embedding='gallery', title='White wine').show()#.savefig('../data/white_whine_2d_gallery.png')
	pairs(X_red, y_red, 'Red wine')
	pairs(X_white, y_white, 'White wine')

###############################
# Classification 
# Prepare data 
###############################
#X = X_white
#y = y_white
X = X_red
y = y_red
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, random_state=23, test_size=0.2)

###############################
# Classify on untransformed dataset. 
###############################
pipeline = Pipeline([('scale', StandardScaler()), ('trans', PCA()), ('cls', GaussianNB())])
cls_nb = WineClassifier(X_train, y_train, X_holdout, y_holdout, pipeline, param_grid={'trans__n_components': np.arange(2,X_train.shape[1]+1, 10)})
cls_nb.train(verbose=1, n_jobs=-1)
cls_nb.classification_report()

pipeline = Pipeline([('scale', StandardScaler()), ('trans', PCA()), ('nn', KNeighborsClassifier())])
cls_nn = WineClassifier(X_train, y_train, X_holdout, y_holdout, pipeline, param_grid={'trans__n_components': np.arange(2,X_train.shape[1]+1, 10), 'nn__n_neighbors': [1, 2, 4, 8, 32, 64]})
cls_nn.train(verbose=1, n_jobs=1) # crashes with n_jobs > 1
cls_nn.classification_report()

pipeline = Pipeline([('scale', StandardScaler()), ('trans', PCA()), ('svc', LinearSVC())])
cls_svc = WineClassifier(X_train, y_train, X_holdout, y_holdout, pipeline, param_grid={'trans__n_components': np.arange(2,X_train.shape[1]+1, 10), 'svc__C': 10. ** np.arange(-3, 4)})
cls_svc.train(verbose=1, n_jobs=1)
cls_svc.classification_report()

pipeline = Pipeline([('scale', StandardScaler()), ('trans', PCA()), ('logistic', LogisticRegression(multi_class='multinomial', solver='lbfgs'))])
cls_log = WineClassifier(X_train, y_train, X_holdout, y_holdout, pipeline, param_grid={'trans__n_components': np.arange(2,X_train.shape[1]+1, 10), 'logistic__C': 10. ** np.arange(-3, 4)})
cls_log.train(verbose=1, n_jobs=1) # Not sure why, but multi_class logisitc regression crashes with multithreading. 
cls_log.classification_report()