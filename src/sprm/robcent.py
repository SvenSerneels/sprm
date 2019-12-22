#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 4 2018
Updated on Sun Dec 16 2018
Refactored on Sat Dec 21 2019

Class for robust centering and scaling of input data for regression and machine
learning 

Version 2.0: Code entirely restructured compared to version 1.0. 
Code made consistent with sklearn logic: fit(data,params) yields results. 
Code makes more effciient use of numpy builtin estimators.
Version 3.0:
Code now takes strings or functions as input to centring and scaling. 
Utility functions have been moved to _preproc_utilities.py 
Code now supplied for l1median cetring, with options to use different 
scipy.optimize optimization algorithms

Parameters
----------
    `center`: str or callable, location estimator. String has to be name of the 
            function to be used, or 'None'. 
    `scale`: str or callable, scale estimator. 

Methods
-------
    `fit(X,trimming)`: Will scale X using 'center' and 'scale' estimators, with 
        a certain trimming fraction when applicable. 
            
Arguments for methods: 
    `X`: array-like, n x p, the data.
    `trimming`: float, fraction to be trimmed (must be in (0,1)). 
    
Ancillary functions in _preproc_utilities.py:
`scale_data(X,m,s)`: centers and scales X on center m (as vector) and 
            scale s (as vector).
`mean(X,trimming)`: Column-wise mean.
`median(X)`: Column-wise median.
`l1median(X)`: L1 or spatial median. Optional arguments: 
    `x0`: starting point for optimization, defaults to column wise median  
    `method`: optimization algorithm, defaults to 'SLSQP' 
    `tol`: tolerance, defaults to 1e-8
    `options`: list of options for `scipy.optimize.minimize`
`std(X,trimming)`: Column-wise std.
`mad(X,c)`: Column-wise median absolute deviation, with consistency factor c. 

             
Remarks
-------
Options for classical estimators 'mean' and 'std' also give access to robust 
trimmed versions.

@author: Sven Serneels, Ponalytics
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition
import numpy as np
from ._m_support_functions import MyException
from ._preproc_utilities import *

class robcent(_BaseComposition,BaseEstimator):
    
    def __init__(self,center='mean',scale='std'):
        
        """
        Initialize values. Check if correct options provided. 
        """
        
        self.center = center
        self.scale = scale 
        
    
    def fit(self,X,**kwargs):
        
        """
        Data standardization according to class' center and scale settings. 
        Trimming fraction can be provided as keyword argument.
        """
        
        if type(self.center) is str: 
            center = eval(self.center)
        else:
            center = self.center
            
        if type(self.scale) is str: 
            scale = eval(self.scale)
        else:
            scale = self.scale
            

        n = X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0] 
            
        if 'trimming' not in kwargs:
            trimming = 0
        else:
            trimming = kwargs.get('trimming')
            
        if self.center == "None":
            m = np.repeat(0,p)
        else:
            m = center(X,trimming=trimming)
            
        setattr(self,"col_loc_",m)    
            
        if self.scale == "None":
            s = np.repeat(1,p)
        else:
            s = scale(X,trimming=trimming)
            
        setattr(self,"col_sca_",s)
            
        Xs = scale_data(X,m,s)
        setattr(self,'datas_',Xs)
            
        return Xs
    
    def predict(self,Xn):
        
        """
        Standardize data on previously estimated location and scale. 
        Number of columns needs to match.
        """
        
        Xns = scale_data(Xn,self.col_loc_,self.col_sca_)
        setattr(self,'datans_',Xns)
        return(Xns)
        