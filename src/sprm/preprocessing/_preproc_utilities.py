#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 10:55:24 2019

Set of help functions for robust centring and scaling 

@author: Sven Serneels, Ponalytics
"""

import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
from statsmodels import robust as srs
import copy

def _handle_zeros_in_scale(scale, copy=True):
    ''' 
    Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    Taken from ScikitLearn.preprocesssing'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale
    
def _check_trimming(t): 
    
    if ((t > .99) or (t < 0)): 
        raise(ValueError("Trimming fraction must be in [0,1)"))
    
def _check_input(X): 
    
    if(type(X) == np.ndarray): 
        X = np.matrix(X)
    
    n,p = X.shape 
    
    if n==1:
        if p > 2: 
            X = X.reshape((-1,1))
        else: 
            raise(ValueError("Statistics not meaningful with fewer than 3 cases"))
    return(X)
    

def mad(X,c=0.6744897501960817,**kwargs):
        
    """
    Column-wise median absolute deviation. **kwargs included to allow 
    general function call in scale_data. 
    """
        
    s = np.median(np.abs(X - np.median(X,axis=0)),axis=0)/c
    s = np.array(s).reshape(-1)
        # statsmodels.robust.mad is not as flexible toward matrix input, 
        # sometimes throws a value error in ufunc
    return s

def median(X,**kwargs):
        
    """
    Column-wise median. **kwargs included to allow 
    general function call in scale_data. 
    """
        
    m = np.median(X,axis=0)
    m = np.array(m).reshape(-1)
    return m

def mean(X,trimming=0):
        
    """
    Column-wise mean or trimmed mean. Trimming to be entered as fraction. 
    """
        
    m = sps.trim_mean(X,trimming,0)
    return m

def std(X,trimming=0):
        
    """
    Column-wise standard devaition or trimmed std. 
    Trimming to be entered as fraction. 
    """
        
    if trimming==0:
        s = np.power(np.var(X,axis=0),.5)
        s = np.array(s).reshape(-1)
    else: 
        var = sps.trim_mean(np.square(X - sps.trim_mean(X,trimming,0)),
                            trimming,0)
        s = np.sqrt(var)   
    return s

def _euclidnorm(x):
    
        
    """
    Euclidean norm of a vector
    """
    
    return(np.sqrt(np.sum(np.square(x))))
    
def _diffmat_objective(a,X):
    
    """
    Utility to l1median, matrix of differences
    """
    
    (n,p) = X.shape
    return(X - np.tile(a,(n,1)))

def _l1m_objective(a,X,*args):
    
    """
    Optimization objective for l1median
    """
    
    return(np.sum(np.apply_along_axis(_euclidnorm,1,_diffmat_objective(a,X))))
    
def _l1m_jacobian(a,X):
    
    """
    Jacobian for l1median
    """
    
    (n,p) = X.shape
    dX = _diffmat_objective(a,X)
    dists = np.apply_along_axis(_euclidnorm,1,dX)
    dX /= np.tile(np.matrix(dists).reshape(n,1),(1,p))
    return(-np.sum(dX,axis=0))

def _l1median(X,x0, method='SLSQP',tol=1e-8,options={'maxiter':2000},**kwargs): 
    
    """
    Optimization for l1median
    """
    
    mu = spo.minimize(_l1m_objective,x0,args=(X),
                      jac=_l1m_jacobian,tol=tol,options=options,method=method)
    return(mu)
    
    
def l1median(X,**kwargs): 
    
    """
    l1median wrapper to generically convert matrices as some of the scipy 
    optimization options will crash when provided matrix input. 
    """
    
    if 'x0' not in kwargs: 
        x0 = np.median(X,axis=0)
        x0 = np.array(x0.T).reshape(-1)

    if type(X) == np.matrix: 
        X = np.array(X)
        
    if len(X.shape) == 2:
        (n,p) = X.shape
    else:
        p = 1
    
    if p<2: 
        return(median(X))
    else:
        return(_l1median(X,x0,**kwargs).x)
        
def kstepLTS(X, maxit = 5, tol = 1e-10,**kwargs):
    
    """
    Computes the K-step LTS estimator of location
    It uses the spatial median as a starting value, and yields an 
    estimator with improved statistical efficiency, but at a higher 
    computational cost. 
    Inputs:
        X: data matrix
        maxit: maximum number of iterations
        tol: convergence tolerance
    Outputs:
        m2: location estimate
    """
    n,p = X.shape
    m1 = l1median(X) # initial estimate
    m2 = copy.deepcopy(m1)
    iteration = 0
    unconverged = True
    while(unconverged and (iteration < maxit)):
        dists = np.sum(np.square(X-m1),axis=1)
        cutdist = np.sort(dists,axis=0)[int(np.floor((n + 1) / 2))-1,0]
        hsubset = np.where(dists <= cutdist)[0]
        m2 = np.array(np.mean(X[hsubset,:],axis=0)).reshape((p,))
        unconverged = (max(abs(m1 - m2)) > tol)
        iteration += 1
        m1 = copy.deepcopy(m2)
    return(m2)
    
def scale_data(X,m,s):
        
        """
        Column-wise data scaling on location and scale estimates. 
        
        """
        
        n = X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]
        
        s = _handle_zeros_in_scale(s)
        
        if p == 1:
            Xm = X - float(m)
            Xs = Xm / s
        else:
            Xm = X - np.matrix([m for i in range(1,n+1)])
            Xs = Xm / np.matrix([s for i in range(1,n+1)])
        return(Xs)       

        
