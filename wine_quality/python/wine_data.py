__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

import pandas as pd
import numpy as np

class WineData(object):
	"""docstring for WineData"""
	def __init__(self, path_to_red, path_to_white):
		self.path_to_red = path_to_red
		self.path_to_white = path_to_white
		
	def _load(self, path_to_data): 
		"""
		Loads the data from data 
		"""
		data = np.array(pd.read_csv(path_to_data, header=0, sep=';'))
		X = data[:,:-1]
		y = data[:,-1]
		return X, y 

	def load_red(self):
		"""
		Loads the red wine data 
		"""
		return self._load(self.path_to_red)

	def load_white(self):
		"""
		Loads the white wine data 
		"""
		return self._load(self.path_to_white)