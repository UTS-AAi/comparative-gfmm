# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:47:45 2018

@author: Thanh Tung Khuat

Fuzzy membership calculation

"""
import numpy as np

def membership_gfmm(X_l, X_u, V, W, g, oper = 'min'):
    """
    Function for membership calculation
    
        b = membership_gfmm(X_l, X_u, V, W, g, oper)
 
   INPUT
     X_l        Input data lower bounds (a row vector with columns being features)
     X_u        Input data upper bounds (a row vector with columns being features)
     V          Hyperbox lower bounds
     W          Hyperbox upper bounds
     g          User defined sensitivity parameter 
     oper       Membership calculation operation: 'min' or 'prod' (default: 'min')
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    
    yW = W.shape[0]
    onesMat = np.ones((yW, 1))
    violMax = 1 - fofmemb(onesMat * X_u - W, g)
    violMin = 1 - fofmemb(V - onesMat * X_l, g)

    if oper == 'prod':
        b = np.prod(np.minimum(violMax, violMin), axis = 1)
    else:
        b = np.minimum(violMax, violMin).min(axis = 1)
    
    return b
  
    
def fofmemb(x, gama):

    """
    fofmemb - ramp threshold function for fuzzy membership calculation

        f = fofmemb(x,gama)
  
   INPUT
     x			Input data matrix (rows = objects, columns = attributes)
     gama		Steepness of membership function
  
   OUTPUT
     f			Fuzzy membership values

   DESCRIPTION
    	f = 1,     if x*gama > 1
    	x*gama,    if 0 =< x*gama <= 1
    	0,         if x*gama < 0
    """

    if np.size(gama) > 1: 
        p = x*(np.ones((x.shape[0], 1))*gama)
    else:
        p = x*gama

    f = ((p >= 0) * (p <= 1)).astype(np.float32) * p + (p > 1).astype(np.float32)
    
    return f

def simpson_membership(Xh, V, W, g):
    """
    Function for membership calculation
    
        b = simpson_membership(Xh, V, W, g)
 
   INPUT
     Xh         Input data (a row vector with columns being features)
     V          Hyperbox lower bounds
     W          Hyperbox upper bounds
     g          User defined sensitivity parameter 
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. This function uses the Simpson's method. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    yW, xW = W.shape
    Xh_mat = np.ones((yW, 1)) * Xh
    zeros_mat = np.zeros((yW, xW))
    
    violMax1 = np.maximum(zeros_mat, 1 - np.maximum(zeros_mat, simpson_membership_min(Xh_mat - W, g)))
    violMax2 = np.maximum(zeros_mat, 1 - np.maximum(zeros_mat, simpson_membership_min(V - Xh_mat, g)))
    
    violMat = (violMax1 + violMax2)
    
    b = np.sum(violMat, axis=1) / (2 * xW)
    
    return b


def simpson_membership_min(x, gamma):
    """
    Min function for fuzzy membership calculation

        f = simpson_membership_min(x, gamma)
  
   INPUT
     x			Input data matrix (rows = objects, columns = attributes)
     gamma	Steepness of membership function
  
   OUTPUT
     f			function's values

   DESCRIPTION
    	f = gamma * min(1, x)
        
    """
    yX, xX = x.shape
    
    if np.size(gamma) > 1: 
        f = (np.ones((yX, 1)) * gamma) * np.minimum(np.ones((yX, xX)), x)
    else:
        f = gamma * np.minimum(np.ones((yX, xX)), x)

    return f
    
    
def asym_similarity_one_many(Xl_k, Xu_k, V, W, g = 1, asym_oper = 'max', oper_mem = 'min'):
    """
    Calculate the asymetrical similarity value of the k-th hyperbox (lower bound - Xl_k, upper bound - Xu_k) and 
    hyperboxes having lower and upper bounds stored in two matrix V and W respectively
    
    INPUT
        Xl_k        Lower bound of the k-th hyperbox
        Xu_k        Upper bound of the k-th hyperbox
        V           Lower bounds of other hyperboxes
        W           Upper bounds of other hyperboxes
        g           User defined sensitivity parameter 
        asym_oper   Use 'min' or 'max' (default) to compute the asymetrical similarity value
        oper_mem    operator used to compute the membership value, 'min' or 'prod'
        
    OUTPUT
        b           similarity values of hyperbox k with all hyperboxes having lower and upper bounds in V and W
    
    """
    numHyperboxes = W.shape[0]
    
    Vk = np.tile(Xl_k, [numHyperboxes, 1])
    Wk = np.tile(Xu_k, [numHyperboxes, 1])
    
    violMax1 = 1 - fofmemb(Wk - W, g)
    violMin1 = 1 - fofmemb(V - Vk, g)
    
    violMax2 = 1 - fofmemb(W - Wk, g)
    violMin2 = 1 - fofmemb(Vk - V, g)
    
    if oper_mem == 'prod':
        b1 = np.prod(np.minimum(violMax1, violMin1), axis = 1)
        b2 = np.prod(np.minimum(violMax2, violMin2), axis = 1)
    else:
        b1 = np.minimum(violMax1, violMin1).min(axis = 1)
        b2 = np.minimum(violMax2, violMin2).min(axis = 1)
        
    if asym_oper == 'max':
        b = np.maximum(b1, b2)
    else:
        b = np.minimum(b1, b2)
    
    return b
