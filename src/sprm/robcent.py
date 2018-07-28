#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:23:15 2018

Class for robust centering and scaling of input data for regression and machine
learning 

Input X to delivered as numerical vector or 2D-array 
Remark: when supplied as np.matrix, functions involving np.median will crash.  

Methods: 
    mean - colwise mean, appended to object as "col_mean"
    median - colwise median, appended to object as "col_med"
    std - colwise std, appended to object as "col_std"
    mad - colwise median absolute deviation (consistency factor c), 
          appended to object as "col_mad"
    daprpr - centers and scales object.X according to cpars=[center,scale] with 
             center and scale the names of one of the functions listed above, 
             or "None". Results of centered and scaled data appended as Xm and
             Xs, and latest location and scale appended as "col_loc" and 
             "col_sca" 

@author: Sven Serneels, Ponalytics
"""

import pandas as ps
import numpy as np

class robcent:
    
    def __init__(self,X=0):
        self.X = X
        
    def mad(self,c=1.4826):
        n = self.X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]
        if not(hasattr(self,"col_med")):
            m = np.apply_along_axis(np.median,0,self.X)
        else: 
            m = self.col_med
        if p==1:
            # Xm = np.matrix(X - m).T
            # s = np.median(abs(Xm))
            # while np.median is broken: 
            Xmf = ps.DataFrame(abs(self.X - m))
            s = Xmf.aggregate(np.median,axis=0)*c
        else:
            Xm = self.X - np.matrix([m for i in range(1,n+1)])
            s = np.apply_over_axes(np.median,np.apply_along_axis(abs,0,Xm),0)*c
        if not(hasattr(self,"col_med")):
            setattr(self,"col_med",m)
        setattr(self,"col_mad",s[0])
        return s[0]
    
    def median(self):
        n = self.X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]
        if not(hasattr(self,"col_med")):
            m = np.apply_along_axis(np.median,0,self.X)
            setattr(self,"col_med",m)
        else: 
            m = self.col_med
        return m
    
    def mean(self):
        n = self.X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]
        if not(hasattr(self,"col_mean")):
            m = np.apply_along_axis(np.mean,0,self.X)
            setattr(self,"col_mean",m)
        else: 
            m = self.col_mean
        return m
    
    def std(self):
        n = self.X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]
        if not(hasattr(self,"col_std")):
            s = np.apply_along_axis(np.std,0,self.X)
            setattr(self,"col_std",s)
        else: 
            s = self.col_std
        return s
    
    def daprpr(self,cpars):
        
        n = self.X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0]   
        if ((type(cpars[0])==str) & (type(cpars[1])==str)):
            if cpars[0] == "None":
                m = np.repeat(0,p)
            else:
                m = eval("self." + cpars[0] + "()")
            setattr(self,"col_loc",m)    
            if cpars[1] == "None":
                s = np.repeat(1,p)
            else:
                s = eval("self." + cpars[1] + "()")
            setattr(self,"col_sca",s)
            Xm = []
            Xs = []
            if not((cpars[0] == "None") & (cpars[1] == "None")):
                if p == 1:
                    Xm = self.X - float(m)
                    setattr(self,"Xm",Xm)
                    if not(cpars[1] == "None"):
                        Xs = Xm / float(s)
                        setattr(self,"Xs",Xs)
                else:
                    Xm = self.X - np.matrix([m for i in range(1,n+1)])
                    setattr(self,"Xm",Xm)
                    if not(cpars[1] == "None"):
                        Xs = Xm / np.matrix([s for i in range(1,n+1)])
                        setattr(self,"Xs",Xs)
        else: 
            if p == 1:
                Xm = self.X - float(cpars[0])
                setattr(self,"Xm",Xm)
                Xs = Xm / cpars[1]
                setattr(self,"Xs",Xm)
            else:
                Xm = self.X - np.matrix([cpars[0] for i in range(1,n+1)])
                setattr(self,"Xm",Xm)
                Xs = Xm / np.matrix([cpars[1] for i in range(1,n+1)])
                setattr(self,"Xs",Xs)
        return self