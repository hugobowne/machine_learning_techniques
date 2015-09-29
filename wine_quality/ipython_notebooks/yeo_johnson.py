# -*- coding: utf-8 -*-
"""
Created on Tue Sept  29 2015

@author: hugobowne-anderson
@email: hugobowne at gmail dot com
"""

from scipy import stats
import numpy as np
import pandas as pd
import math as math

def yeo_johnson(x, lmbda=0 ):
    """
    Performs a Yeo-Johnson Transformation on a numpy array.
    Arguments:
    Input array. Should be 1-dimensional.
    lmbda : {scalar}, optional.
    IN PROGRESS: I WILL COMMENT CODE BELOW ASAP; A RUNTIME WARNING MAY BE THROWN
    DURING EXECUTION BUT THE RESULT SHOULD NOT BE AFFECTED.
    I HAVE USED THE DEFINITION OF YEO-JOHNSON TRANSFORMATION FROM THE ORIGINAL PAPER:
    Yeo, In-Kwon and Johnson, Richard (2000). A new family of power transformations
    to improve normality or symmetry. Biometrika, 87, 954-959.
    """
    if lmbda == 0:
        A1 = np.log(abs(x+1)) 
        A1[A1 == -np.inf] = 0
        A2 = (np.power(1-x , 2) - 1)/2
        A2[np.isnan(A2)] = 0
        x_yj = (x>=0)*A1 - (x<0)*A2
    elif lmbda == 2:
        B1 = (np.power(x+1 , 2) - 1)/2
        B1[np.isnan(B1)] = 0
        B2 = np.log(abs(1-x))
        B2[B2==-np.inf] = 0
        x_yj = (x>=0)*B1 - (x<0)*B2
    else:
        C1 = (np.power(x+1 , lmbda) - 1)/lmbda
        C1[np.isnan(C1)] = 0
        C2 = (np.power(1-x , 2-lmbda) - 1)/(2 - lmbda)
        C2[np.isnan(C2)] = 0
        x_yj = (x>=0)*C1 + (x<0)*C2

    return x_yj
    
def dfyeo_johnson(df, lmbda=0 ):
    """
    Performs a Yeo-Johnson Transformation on all columns (features of a dataframe)
    """
    df_yj = pd.DataFrame(columns=df.columns)  #initialize empty data frame with same features as df
    for val in list(df.columns):
        df_yj[val] = yeo_johnson(df[val]) #populate dataframe with transformed data
    return df_yj
