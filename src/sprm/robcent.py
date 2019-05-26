#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 4 2018
Updated on Sun Dec 16 2018

Class for robust centering and scaling of input data for regression and machine
learning 

Version 2.0: Code entirely restructured compared to version 1.0. 
Code made consistent with sklearn logic: fit(data,params) yields results. 
Code makes more effciient use of numpy builtin estimators.

Parameters
----------
    center: str, location estimator. Presently allowed: 'mean', 'median' or
                 'None'. 
    scale: str, scale estimator. Presently allowed: 'mad', 'std' or 'None'. 
                 
    Note that 'sd' also gives access to robust trimmed stds.

Methods
-------
    fit(X,trimming): Will scale X using 'center' and 'scale' estimators. 
    mean(X,trimming): Column-wise mean, appended to object as "col_mean_".
    median(X): Column-wise median, appended to object as "col_med_".
    std(X,trimming): Column-wise std, appended to object as "col_std_".
    mad(X,c):  Column-wise median absolute deviation, 
            appended to object as "col_mad_".
    scale_data(X,m,s) - centers and scales X on center m (as vector) and 
            scale s (as vector).
            
Arguments for methods: 
    X: array-like, n x p, the data.
    trimming: float, fraction to be trimmed (must be in (0,1)).
    c, float, consistency factor.
             
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
import scipy.stats as sps
from statsmodels import robust as srs
from ._m_support_functions import MyException

class robcent(_BaseComposition,BaseEstimator):
    
    def __init__(self,center='mean',scale='std'):
        
        """
        Initialize values. Check if correct options provided. 
        """
        
        self.center = center
        self.scale = scale 
        self.licenter = ['mean','median','None']
        self.liscale = ['mad','std','None']
        if not(self.center in self.licenter):
            raise(MyException('center options are: "mean", "median", "None"'))
        if not(self.scale in self.liscale):
            raise(MyException('scale options are: "mad", "std", "None"'))
        
    def mad(self,X,c=0.6744897501960817,**kwargs):
        
        """
        Column-wise median absolute deviation. **kwargs included to allow 
        general function call in scale_data. 
        """
        
        s = srs.mad(X,c=c,axis=0)
        setattr(self,"col_mad_",s)
        
        return s
    
    def median(self,X,**kwargs):
        
        """
        Column-wise median. **kwargs included to allow 
        general function call in scale_data. 
        """
        
        m = np.median(X,axis=0)
        m = np.array(m).reshape(-1)
        setattr(self,"col_med_",m)
        
        return m
    
    def mean(self,X,trimming=0):
        
        """
        Column-wise mean or trimmed mean. Trimming to be entered as fraction. 
        """
        
        m = sps.trim_mean(X,trimming,0)
        setattr(self,"col_mean_",m)
        
        return m
    
    def std(self,X,trimming=0):
        
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
            
        setattr(self,"col_std_",s)    
        return s
    
    def scale_data(self,X,m,s):
        
        """
        Column-wise data scaling on location and scale estimates. 
        
        """
        
        n = X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]
        
        if p == 1:
            Xm = X - float(m)
            Xs = Xm / s
        else:
            Xm = X - np.matrix([m for i in range(1,n+1)])
            Xs = Xm / np.matrix([s for i in range(1,n+1)])
        return(Xs)
    
    
    def fit(self,X,**kwargs):
        
        """
        Data standardization according to class' center and scale settings. 
        Trimming fraction can be provided as keyword argument.
        """
        
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
            m = eval("self." + self.center + "(X,trimming=trimming)")
            
        setattr(self,"col_loc_",m)    
            
        if self.scale == "None":
            s = np.repeat(1,p)
        else:
            s = eval("self." + self.scale + "(X,trimming=trimming)")
            
        setattr(self,"col_sca_",s)
            
        Xs = self.scale_data(X,m,s)
        setattr(self,'datas_',Xs)
            
        return Xs
    
    def predict(self,Xn):
        
        """
        Standardize data on previously estimated location and scale. 
        Number of columns needs to match.
        """
        
        Xns = self.scale_data(Xn,self.col_loc_,self.col_sca_)
        setattr(self,'datans_',Xns)
        return(Xns)
        