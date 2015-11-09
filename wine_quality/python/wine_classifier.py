__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report

class WineClassifier(object):
	"""
	Use classification (not regression) for wine quality. 
	"""
	def __init__(self, X_train, y_train, X_valid, y_valid, pipeline, param_grid):
		"""
		Set the data sets. 
		"""
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.pipeline = pipeline
		self.param_grid = param_grid

	def train(self, verbose=1, n_jobs=-1, scoring='accuracy', cv=10): 
		"""
		Train the classifier by grid search 
		"""
		if len(self.param_grid) != 0: 
			grid_search = GridSearchCV(self.pipeline, param_grid=self.param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs, scoring=scoring)
			grid_search.fit(self.X_train, self.y_train)
			if verbose > 1: 
				print( ('Best score %s with parameters %s') % (grid_search.best_score_, grid_search.best_params_))
			self.pipeline = grid_search.best_estimator_
		else: 
			if verbose > 1:
				scores = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=cv)
				print(('Best score %s') % (scores.mean()))
			self.pipeline.fit(self.X_train, self.y_train)

	def classification_report(self, print_stdout=True):
		"""
		Valid classifier on validation set. 
		"""
		report = classification_report(self.y_valid, self.pipeline.predict(self.X_valid))
		if print_stdout: print(report)
		return(report)





		