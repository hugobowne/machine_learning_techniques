__author__ = "Fernando Carrillo"
__email__ = "fernando at carrillo.at"

from wine_data import WineData
from wine_explore import plot2d, pairs

from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

data = WineData('../winequality-red.csv', '../winequality-white.csv')

X_red, y_red = data.load_red()
X_white, y_white = data.load_white()

###############################
# Explore data 
# 1. Plot in 2d, color code classes 
#	-> few very strong outliers: consider removing. 
# 2. Plot paris 
#	-> correlation: transform data or use regularized methods
#	-> non-normal distributed featues: Box-Cox transform
###############################
#pairs(X_red, y_red, 'Red wine')
#pairs(X_white, y_white, 'White wine')

# Plot in 2d. Both have strong outliers. Consider removing. 
plot2d(X_red, y_red, scale=True, embedding='pca', title='Red wine')
plot2d(X_red, y_red, scale=True, embedding='isomap', title='Red wine')
plot2d(X_red, y_red, scale=True, embedding='lle', title='Red wine')
plot2d(X_red, y_red, scale=True, embedding='spectral', title='Red wine')
plot2d(X_red, y_red, scale=True, embedding='mds', title='Red wine')
plot2d(X_red, y_red, scale=True, embedding='tsne', title='Red wine')

#plot2d(X_white, y_white, 'White wine')