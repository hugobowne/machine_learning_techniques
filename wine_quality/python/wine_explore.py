__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, SpectralEmbedding, MDS

import pandas as pd

def plot2d(X, y, scale=True, normalize=False, embedding='pca', title=''): 
	"""
	Plot data transformed into two dimensions by PCA. 
	PCA transforms into a new embedding dimension such that 
	the first dimension contains the maximal variance and following 
	dimensions maximal remaining variance. 
	This shoudl spread the observed n-dimensional data maximal. This 
	is unsupervised and will not consider target values. 
	"""
	if (scale): 
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

	if (normalize): 
		normalizer = Normalizer(norm='l2')
		X = normalizer.fit_transform(X)
		
	if (embedding is 'pca'): 
		pca = PCA(n_components=2)
		X_transformed = pca.fit_transform(X)
	elif (embedding is 'isomap'):
		isomap = Isomap(n_components=2, n_neighbors=20)
		X_transformed = isomap.fit_transform(X)
	elif (embedding is 'lle' ): 
		lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
		X_transformed = lle.fit_transform(X)
	elif (embedding is 'tsne'): 
		t_sne = TSNE(n_components=2)
		X_transformed = t_sne.fit_transform(X)
	elif (embedding is 'spectral'): 
		se = SpectralEmbedding(n_components=2)
		X_transformed = se.fit_transform(X)
	elif (embedding is 'mds'):
		mds = MDS(n_components=2)
		X_transformed = mds.fit_transform(X)
	elif (embedding is 'gallery'): 
		plt.figure(1)
		
		plt.subplot(231)
		plt.title('pca')
		X_t = PCA(n_components=2).fit_transform(X)
		plt.scatter(X_t[:,0 ], X_t[:, 1], c=y)

		plt.subplot(232)
		plt.title('isomap')
		X_t = Isomap(n_neighbors=20).fit_transform(X)
		plt.scatter(X_t[:,0 ], X_t[:, 1], c=y)

		plt.subplot(233)
		plt.title('lle')
		X_t = LocallyLinearEmbedding(n_neighbors=20).fit_transform(X)
		plt.scatter(X_t[:,0 ], X_t[:, 1], c=y)

		plt.subplot(234)
		plt.title('tsne')
		X_t = TSNE().fit_transform(X)
		plt.scatter(X_t[:,0 ], X_t[:, 1], c=y)

		plt.subplot(235)
		plt.title('spectral')
		X_t = SpectralEmbedding().fit_transform(X)
		plt.scatter(X_t[:,0 ], X_t[:, 1], c=y)

		plt.subplot(236)
		plt.title('mds')
		X_t = MDS().fit_transform(X)
		plt.scatter(X_t[:,0 ], X_t[:, 1], c=y)

		plt.suptitle('Gallery transforms ' + title)

		return plt
	else:
		raise ValueError("Choose between pca, isomap and tsne")

	plt.title(title + ' ' + embedding + ' plot')
	sc = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
	plt.colorbar(sc)
	return plt

def plot2d_gallery(self, X, y, scale=True): 
	if (scale): 
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

	plt.figure(1)
	plt.subplot(231)
	plt.axes.append()

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
