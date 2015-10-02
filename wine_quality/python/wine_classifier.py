__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

class WineClassifier(object):
	"""
	Use classification (not regression) for wine quality. 
	"""
	def __init__(self, arg):
		super(WineClassifier, self).__init__()
		self.arg = arg
		