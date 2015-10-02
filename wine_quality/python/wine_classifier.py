__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

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
	




		