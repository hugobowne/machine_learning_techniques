__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

import pandas as pd

def plot2d(X, y, embedding='pca', title=''): 
	"""
	Plot data transformed into two dimensions by PCA. 
	PCA transforms into a new embedding dimension such that 
	the first dimension contains the maximal variance and following 
	dimensions maximal remaining variance. 
	This shoudl spread the observed n-dimensional data maximal. This 
	is unsupervised and will not consider target values. 
	"""
	if (embedding is 'pca'): 
		pca = PCA(n_components=2)
		pca.fit(X)
		X_transformed = pca.transform(X)
	elif (embedding is 'isomap'):
		isomap = Isomap(n_components=2, n_neighbors=20)
		X_transformed = isomap.fit_transform(X)
	elif (embedding is 'tsne'): 
		t_sne = TSNE(n_components=2)
		X_transformed = t_sne.fit_transform(X)
	else:
		raise ValueError("Choose between pca, isomap and tsne")

	plt.title(title + ' ' + embedding + ' plot')
	sc = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
	plt.colorbar(sc)
	plt.show()

def pairs(X, y, title): 
	"""
	Quick and dirty version of pairs. 
	"""
	df = pd.DataFrame(X)
	df[df.shape[1]] = y
	plt.title(title + ' Pairwise plot')
	axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
	#plt.tight_layout()
	#plt.savefig('scatter_matrix.png')
	plt.show() 
