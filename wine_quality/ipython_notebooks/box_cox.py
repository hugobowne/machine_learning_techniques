# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:20:50 2015

@author: hugobowne-anderson
@email: hugobowne at gmail dot com
"""

from scipy import stats
import pandas as pd

def box_cox(df, lmbda=None, alpha=None):
    """
    Performs a Box-Cox Transformation on all columns (features) of a pandas
    dataframe. Currently, there is some ambiguity as to how to deal with
    non-positive values & I need to check this out: at the moment, I just centre
    the data so that min(value) > 0, for all features, as necessitated by
    the very nature of the Box-Cox Transformation.
    """
    df_tr = pd.DataFrame(columns=df.columns)  #initialize empty data frame with same features as df
    for val in list(df.columns):
        df_tr[val] = stats.boxcox(df[val] - min(df[val]) + 0.1,lmbda, alpha)[0] #populate dataframe with transformed data
    return df_tr
