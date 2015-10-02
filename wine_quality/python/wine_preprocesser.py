__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

import numpy as np 
from sklearn.preprocessing import PolynomialFeatures

class WinePreprocesser(object):
	"""docstring for WinePreprocesser"""
	def __init__(self, wine_data):
		self.X_red, self.y_red = wine_data.load_red()
		self.X_white, self.y_white = wine_data.load_white()

	def _divide_features(self, X, replace_inf_with_absmax): 
		"""
		Divide 1 by feature value. 
		"""
		# Do the division 
		nf = np.divide(1, X)
		# Replace inf by nan or by the maximal absolute value 
		for i in np.arange(nf.shape[1]):
			if np.inf in nf[:,i]: 
				a = nf[:,i]
				if replace_inf_with_absmax: 
					a[np.isinf(a)] = a[np.argmax(abs(a[np.isfinite(a)]))]
				else:
					a[np.isinf(a)] = np.nan
				nf[:,i] = a 
		return(nf)
		

	def add_divided_features(self, replace_inf_with_absmax=True):
		"""
		For each feature y_i add 1/y_i
		"""
		X_red_divided = self._divide_features(X=self.X_red, replace_inf_with_absmax=replace_inf_with_absmax)
		self.X_red = np.concatenate((self.X_red, X_red_divided), axis=1)
		X_white_divided = self._divide_features(X=self.X_white, replace_inf_with_absmax=replace_inf_with_absmax)
		self.X_white = np.concatenate((self.X_white, X_white_divided), axis=1)		

	def polynomial_expansion(self, rank=2): 
		"""
		Expand the features with polynonial of rank rnank 
		"""
		pf = PolynomialFeatures(degree=2)
		self.X_red = pf.fit_transform(self.X_red)
		self.X_white = pf.fit_transform(self.X_white)

	def _remove_low_var(self, X, variance_threshold): 
		"""
		Remove features with variance below threshold. 
		"""
		remove_index = [] 
		for col in range(X.shape[1]): 
			if np.var(X[:,col]) < variance_threshold: 
				remove_index.append(col)
		return(np.delete(X, remove_index, 1)) 

	def remove_low_variance_features(self, variance_threshold=0): 
		"""
		Remove features with variance below threshold. 
		"""
		self.X_red = self._remove_low_var(self.X_red, variance_threshold)
		self.X_white = self._remove_low_var(self.X_white, variance_threshold)

	def yeo_johnson_transform(self): 
		"""
		Implement yeo johnson transform 
		"""
		raise NotImplementedError

	def get_red(self):
		"""
		Returns X, y of red wine data 
		"""
		return self.X_red, self.y_red 

	def get_white(self):
		"""
		Returns X, y of white wine data 
		"""
		return self.X_white, self.y_white 
		