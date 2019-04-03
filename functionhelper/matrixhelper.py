# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:50:53 2018

@author: Thanh Tung Khuat

Helper function for matrix handling
"""

from sklearn.decomposition import PCA
import numpy as np

def pca_transform(A, n):
    """
    Perform Principal Component Analysis for the input maxtrix to reduce the current dimensions into
    new dimensions, n
    
    INPUT:
        A   Input maxtrix
        n   new dimensions
    
    OUTPUT:
        New n-dimensional matrix
    """
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(A)
    
    return principalComponents

def delete_const_dims(Xl, Xu = None):
    """
    delete any constant dimensions (the same value for all samples) within both lower and upper bounds
    
          Xl, Xu = delete_const_dims(Xl, Xu)      
    
    INPUT:
        Xl      Lower bounds matrix (rows: objects, cols: features)
        Xu      Upper bounds matrix (rows: objects, cols: features)
        
    OUTPUT:
        Xl, Xu after deleting all constant dimensions
        
    If input is only one matrix, the operation is performed for that matrix
    """
    numDims = Xl.shape[1]
    colMask = np.array([True] * numDims)
	
    for i in range(numDims):
        if (Xl[:, i] == Xl[0, i]).all() == True and ((Xu is not None and (Xu[:, i] == Xu[0, i]).all() == True) or (Xu is None)):
            colMask[i] = False
    
    Xl_n = Xl[:, colMask]
    
    if Xu is not None:
        Xu_n = Xu[:, colMask]
        
        return (Xl_n, Xu_n)
    else:
        return Xl_n