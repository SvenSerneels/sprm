#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:56:12 2018

Class containing:
    Sparse Partial Robust M Regression (SPRM)
    Sparse NIPALS (SNIPLS) 
    Ancillary functions: 
        Fair function
        Huber function
        Hampel function 
        Broken Stick function

Depends on robcent class for robustly centering and scaling data 

@author: Sven Serneels, Ponalytics
"""
import numpy as np
import pandas as ps
from scipy.stats import norm, chi2
from sklearn.base import RegressorMixin,BaseEstimator,TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
import copy

class MyException(Exception):
        pass



class snipls(_BaseComposition,BaseEstimator,TransformerMixin,RegressorMixin):
    """
    SNIPLS Sparse Nipals Algorithm 
    
    Algorithm first outlined in: 
        Sparse and robust PLS for binary classification, 
        I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, 
        Journal of Chemometrics, 30 (2016), 153-162.
    
    Parameters:
    -----------
    eta: float. Sparsity parameter in [0,1)
    n_components: int, min 1. Note that if applied on data, n_components shall 
        take a value <= min(x_data.shape)
    verbose: Boolean (def true): to print intermediate set of columns retained
    colums (def false): Either boolean or list
        if False, no column names supplied 
        if a list (will only take length x_data.shape[1]), the column names of 
            the x_data supplied in this list, will be printed in verbose mode
    copy (def True): boolean, whether to copy data
    Note: copy not yet aligned with sklearn def  - we always copy    
    
    """
    
    def __init__(self,eta=.5,n_components=1,verbose=True,columns=False,copy=True):
        self.eta = eta 
        self.n_components = n_components
        self.verbose = verbose
        self.columns = columns
        self.copy = copy

    def fit(self,X,y):
        if self.copy:
            self.X = X
            self.y = y
        X = X.astype("float64")
        (n,p) = X.shape
        y = y.astype("float64")
        Xh = robcent(X)
        Xh.daprpr(["mean","None"])
        X0 = Xh.Xm
        yh = robcent(y)
        yh.daprpr(["mean","None"])
        y0 = yh.Xm
        T = np.empty((n,self.n_components),float) 
        W = np.empty((p,self.n_components),float)  
        P = np.empty((p,self.n_components),float)
        C = np.empty((self.n_components,1),float) 
        Xev = np.empty((self.n_components,1),float)
        yev = np.empty((self.n_components,1),float)
        B = np.empty((p,1),float) 
        oldgoodies = np.array([])
        Xi = Xh.Xm
        yi = np.matrix(yh.Xm).T
        for i in range(1,self.n_components+1):
            wh =  Xi.T * yi
            wh = wh/np.linalg.norm(wh,"fro")
            # goodies = abs(wh)-llambda/2 lambda definition
            goodies = abs(wh)-self.eta*max(abs(wh))
            wh = np.multiply(goodies,np.sign(wh))
            goodies = np.where((goodies>0))[0]
            goodies = np.union1d(oldgoodies,goodies)
            oldgoodies = goodies
            if len(goodies)==0:
                print("No variables retained at" + str(i) + "latent variables" +
                      "and lambda = " + str(self.eta) + ", try lower lambda")
                break
            elimvars = np.setdiff1d(range(0,p),goodies)
            wh[elimvars] = 0 
            th = Xi * wh
            nth = np.linalg.norm(th,"fro")
            ch = (yi.T * th)/(nth**2)
            ph = (Xi.T * Xi * wh)/(nth**2)
            ph[elimvars] = 0 
            yi = yi - th*ch 
            W[:,i-1] = np.reshape(wh,p)
            P[:,i-1] = np.reshape(ph,p)
            C[i-1] = ch 
            T[:,i-1] = np.reshape(th,n)
            Xi = Xi - th * ph.T
            Xev[i-1] = (nth**2*np.linalg.norm(ph,"fro")**2)/np.sum(np.square(X0))*100
            yev[i-1] = np.sum(nth**2*(ch**2))/np.sum(y0**2)*100
            if type(self.columns)==bool:
                colret = goodies
            else:
                colret = self.columns[np.setdiff1d(range(0,p),elimvars)]
            if(self.verbose):
                print("Variables retained for " + str(i) + " latent variable(s):" +
                      "\n" + str(colret) + ".\n")
        if(len(goodies)>0):
            R = np.matmul(W[:,range(0,i)] , np.linalg.inv(np.matmul(P[:,range(0,i)].T,W[:,range(0,i)])))
            B = np.matmul(R,C[range(0,i)])
        else:
            B = np.empty((p,1))
            B.fill(0)
            R = B
            T = np.empty((n,n_components))
            T.fill(0)
        yp = np.array(X0 * B + yh.col_loc).astype("float64")
        setattr(self,"x_weights_",W)
        setattr(self,"x_loadings_",P)
        setattr(self,"C_",C)
        setattr(self,"x_scores_",T)
        setattr(self,"coef_",B)
        setattr(self,"intercept_",np.mean(y0 - np.matmul(X0,B)))
        setattr(self,"x_ev_",Xev)
        setattr(self,"y_ev_",yev)
        setattr(self,"fitted_",yp.reshape(-1))
        setattr(self,"x_Rweights_",R)
        setattr(self,"colret_",colret)
        setattr(self,"x_loc_",Xh.col_loc)
        setattr(self,"y_loc_",yh.col_loc)
        setattr(self,"x_sca_",Xh.col_sca)
        setattr(self,"y_sca_",yh.col_sca)
        return(self)
        
    
    def predict(self,Xn):
        Xnc = robcent(Xn)
        Xnc.daprpr([self.x_loc_,self.x_sca_])
        return(np.matmul(Xnc.Xm,self.coef_) + self.y_loc_)



class sprm(_BaseComposition,BaseEstimator,TransformerMixin,RegressorMixin):
    
    """
    SPRM Sparse Partial Robust M Regression 
    
    Algorithm first outlined in: 
        Sparse partial robust M regression, 
        Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, 
        Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59. 
    
    Parameters:
    -----------
    eta: float. Sparsity parameter in [0,1)
    n_components: int, min 1. Note that if applied on data, n_components shall 
        take a value <= min(x_data.shape)
    fun: str, downweighting function. 'Hampel' (recommended), 'Fair' or 
                'Huber'
    probp1: float, probability cutoff for start of downweighting 
                 (e.g. 0.95)
    probp2: float, probability cutoff for start of steep downweighting 
                 (e.g. 0.975, only relevant if fun='Hampel')
    probp3: float, probability cutoff for start of outlier omission 
                 (e.g. 0.999, only relevant if fun='Hampel')
    centring: str, type of centring ('mean' or 'median' [recommended])
    scaling: str, type of scaling ('std','mad' [recommended] or 'None')
    verbose: boolean, specifying verbose mode
    maxit: int, maximal number of iterations in M algorithm
    tol: float, tolerance for convergence in M algorithm 
    start_cutoff_mode: str, values:
        'specific' will set starting value cutoffs specific to X and y (preferred); 
        any other value will set X and y stating cutoffs identically. 
        Even though not preferred, the non-specific setting was included because
        it yields identical results to the SPRM R implementation available from
        CRAN.
    start_X_init: str, values:
        'pcapp' will include a PCA/broken stick projection to 
                calculate the staring weights, else just based on X;
        any other value will calculate the X starting values based on the X
                matrix itself. This is less stable for very flat data (p >> n),
                yet it is the only option implemented in the CRAN R version. 
        
    colums (def false): Either boolean or list
        if False, no column names supplied 
        if a list (will only take length x_data.shape[1]), the column names of 
            the x_data supplied in this list, will be printed in verbose mode
    copy (def True): boolean, whether to copy data
        Note: copy not yet aligned with sklearn def  
    
    """
    
    def __init__(self,n_components=1,eta=.5,fun='Hampel',probp1=0.95
                 ,probp2=0.975,probp3=0.999,centring='median',scaling='mad'
                 ,verbose=True,maxit=100,tol=0.01,start_cutoff_mode='specific'
                 ,start_X_init='pcapp',columns=False,copy=True):
        self.n_components = int(n_components) 
        self.eta = float(eta)
        self.fun = fun
        self.probp1 = probp1
        self.probp2 = probp2
        self.probp3 = probp3
        self.centring = centring
        self.scaling = scaling
        self.verbose = verbose
        self.maxit = maxit
        self.tol = tol
        self.start_cutoff_mode = start_cutoff_mode
        self.start_X_init = start_X_init
        self.columns = columns
        self.copy = copy
        self.probctx_ = 'irrelevant'
        self.probcty_ = 'irrelevant'
        self.hampelbx_ = 'irrelevant'
        self.hampelby__ = 'irrelevant'
        self.hampelrx_ = 'irrelevant'
        self.hampelry_ = 'irrelevant'
        
    def brokenstick(self,n_components):
        q = np.triu(np.ones((n_components,n_components)))
        r = np.empty((n_components,1),float)
        r[0:n_components,0] = (range(1,n_components+1))
        q = np.matmul(q,1/r)
        q /= n_components
        return q
    
    def Fair(self,x,probct):
         return((1/(1 + abs(x/(probct * 2)))**2)) 
         
    def Huber(self,x,probct):
        x[np.where(x <= probct)[0]] = 1
        x[np.where(x > probct)] = probct/abs(x[np.where(x > probct)])
        return(x)
        
    def Hampel(self,x,probct,hampelb,hampelr):
        wx = x
        wx[np.where(x <= probct)[0]] = 1
        wx[np.where((x > probct) & (x <= hampelb))[0]] = probct/abs(x[
                    np.where((x > probct) & (x <= hampelb))[0]])
        wx[np.where((x > hampelb) & (x <= hampelr))[0]] = np.divide(
                probct * (hampelr - (x[np.where((x > hampelb) 
                        & (x <= hampelr))[0]])),
                       (hampelr - hampelb) * abs(x[np.where((x > hampelb) &
                       (x <= hampelr))[0]])
                        )
        wx[np.where(x > hampelr)[0]] = 0
        return(wx)

     #  pars = ["Hampel",0.95, 0.975, 0.999, "median", "mad", False, 100, .01 ]
    def fit(self,X,y):
        if self.copy:
            self.X = copy.deepcopy(X)
            self.y = copy.deepcopy(y)
        (n,p) = X.shape
        if not(type(self.n_components)==int) | (self.n_components<=0):
            raise MyException("Number of components has to be a positive integer")
        if ((self.n_components > n) | (self.n_components > p)):
            raise MyException("The number of components is too large.")
        if (self.n_components <= 0):
            raise MyException("The number of components has to be positive.")
        if not(type(self.eta)==float):
            raise MyException("Sparsity parameter eta has to be a floating point number")
        if ((self.eta < 0) | (self.eta >= 1)):
            raise MyException("eta has to come from the interval [0,1)")
        if (not(self.fun in ("Hampel", "Huber", "Fair"))):
            raise MyException("Invalid weighting function. Choose Hampel, Huber or Fair for parameter fun.")
        if ((self.probp1 > 1) | (self.probp1 <= 0)):
            raise MyException("probp1 is a probability. Choose a value between 0 and 1")
        if (self.fun == "Hampel"):
            if (not((self.probp1 < self.probp2) & (self.probp2 < self.probp3) & (self.probp3 <= 1))):
                raise MyException("Wrong choise of parameters for Hampel function. Use 0<probp1<hampelp2<hampelp3<=1")
        ny = y.shape[0]
        if ny != n:
            raise MyException("Number of cases in y and X must be identical.")

        Xrc = robcent(X)
        yrc = robcent(y)
        Xrc.daprpr([self.centring,self.scaling])
        yrc.daprpr([self.centring,self.scaling])
        #datac <- attr(datamc, "Center")
        #datas <- attr(datamc, "Scale")
        #attr(datac, "Type") <- center
        y0 = y
        ys = yrc.Xs.astype('float64')
        Xs = Xrc.Xs.astype('float64')
        if (self.start_X_init=='pcapp'):
            U, S, V = np.linalg.svd(Xs)
            spc = np.square(S)
            spc /= np.sum(spc)
            relcomp = max(np.where(spc - self.brokenstick(min(p,n))[:,0] <=0)[0][0],1)
            Urc = robcent(np.array(U[:,0:relcomp]))
            Urc.daprpr([self.centring,self.scaling])
        else: 
            Urc = Xrc
        wx = np.sqrt(np.array(np.sum(np.square(Urc.Xs),1),dtype=np.float64))
        wx = wx/np.median(wx)
        if [self.centring,self.scaling]==['median','mad']:
            wy = np.array(abs(yrc.Xs),dtype=np.float64)
        else:
            wy = (y - np.median(y))/(1.4826*np.median(abs(y-np.median(y))))
        self.probcty_ = norm.ppf(self.probp1)
        if self.start_cutoff_mode == 'specific':
            self.probctx_ = chi2.ppf(self.probp1,relcomp)
        else: 
            self.probctx_ = self.probcty_
        if (self.fun == "Fair"):
            wx = self.Fair(wx,self.probctx_)
            wy = self.Fair(wy,self.probcty_)
        if (self.fun == "Huber"):
            wx = self.Huber(wx,self.probctx_)
            wy = self.Huber(wy,self.probcty_)
        if (self.fun == "Hampel"):
            self.hampelby_ = norm.ppf(self.probp2)
            self.hampelry_ = norm.ppf(self.probp3)
            if self.start_cutoff_mode == 'specific':
                self.hampelbx_ = chi2.ppf(self.probp2,relcomp)
                self.hampelrx_ = chi2.ppf(self.probp3,relcomp)
            else: 
                self.hampelbx_ = self.hampelby_
                self.hampelrx_ = self.hampelry_
            wx = self.Hampel(wx,self.probctx_,self.hampelbx_,self.hampelrx_)
            wy = self.Hampel(wy,self.probcty_,self.hampelby_,self.hampelry_)
        wx = np.array(wx).reshape(-1)
        w = (wx*wy).astype("float64")
        if (w < 1e-06).any():
            w0 = np.where(w < 1e-06)
            w[w0] = 1e-06
            we = np.array(w,dtype=np.float64)
        else:
            we = np.array(w,dtype=np.float64)
        wte = wx
        wye = wy
        WEmat = np.array([np.sqrt(we) for i in range(1,p+1)],ndmin=1).T    
        Xw = np.multiply(Xrc.Xs,WEmat).astype("float64")
        yw = ys*np.sqrt(we)
        loops = 1
        rold = 1E-5
        difference = 1
        # Begin at iteration
        while ((difference > self.tol) & (loops < self.maxit)):
            res_snipls = snipls(self.eta,self.n_components,
                            self.verbose,self.columns,self.copy)
            res_snipls.fit(Xw,yw)
            T = np.divide(res_snipls.x_scores_,WEmat[:,0:(self.n_components)])
            b = res_snipls.coef_
            yp = res_snipls.fitted_
            r = ys - yp
            if (len(r)/2 > np.sum(r == 0)):
                r = abs(r)/(1.4826 * np.median(abs(r)))
            else:
                r = abs(r)/(1.4826 * np.median(abs(r[r != 0])))
            scalet = self.scaling
            if (scalet == "None"):
                scalet = "mad"
            dt = robcent(T)
            dt.daprpr([self.centring, scalet])
            wtn = np.sqrt(np.array(np.sum(np.square(dt.Xs),1),dtype=np.float64))
            wtn = wtn/np.median(wtn)
            wtn = wtn.reshape(-1)
            wye = r
            wte = wtn
            if (self.fun == "Fair"):
                wte = self.Fair(wtn,self.probctx_)
                wye = self.Fair(wye,self.probcty_)
            if (self.fun == "Huber"):
                wte = self.Huber(wtn,self.probctx_)
                wye = self.Huber(wye,self.probcty_)
            if (self.fun == "Hampel"):
                self.probctx_ = chi2.ppf(self.probp1,self.n_components)
                self.hampelbx_ = chi2.ppf(self.probp2,self.n_components)
                self.hampelrx_ = chi2.ppf(self.probp3,self.n_components)
                wte = self.Hampel(wtn,self.probctx_,self.hampelbx_,self.hampelrx_)
                wye = self.Hampel(wye,self.probcty_,self.hampelby_,self.hampelry_)
            b2sum = np.sum(b**2)    
            difference = abs(b2sum - rold)/rold
            rold = b2sum
            wte = np.array(wte).reshape(-1)
            we = (wye * wte).astype("float64")
            w0=[]
            if (any(we < 1e-06)):
                w0 = np.where(we < 1e-06)
                we[w0] = 1e-06
                we = np.array(we,dtype=np.float64)
            if (len(w0) >= (n/2)):
                break
            WEmat = np.array([np.sqrt(we) for i in range(1,p+1)],ndmin=1).T    
            Xw = np.multiply(Xrc.Xs,WEmat).astype("float64")
            yw = ys*np.sqrt(we)
            loops += 1
        if (difference > self.maxit):
            print("Warning: Method did not converge. The scaled difference between norms of the coefficient vectors is " + 
                  str(round(difference,4)))
        plotprec = False
        if plotprec:
            print(str(loops - 1))
        w = we
        w[w0] = 0
        wt = wte
        wt[w0] = 0
        wy = wye
        wy[w0] = 0
        P = res_snipls.x_loadings_
        W = res_snipls.x_weights_
        R = res_snipls.x_Rweights_
        Xrw = np.array(np.multiply(Xrc.Xs,np.sqrt(WEmat)).astype("float64"))
        Xrw = robcent(Xrw)
        Xrw.daprpr([self.centring,"None"]) 
        # these four lines likely phantom code 
        # in R implementation multiplication just with centered data 
                
        # T = Xrw.Xm.astype("float64")*R 
        T = Xs * R
        if self.verbose:
            print("Final Model: Variables retained for " + str(n_components) + " latent variables: \n" 
                 + str(res_snipls.colret_) + "\n")
        b_rescaled = np.reshape(yrc.col_sca/Xrc.col_sca,(5,1))*b
        if(self.centring == "mean"):
            intercept = np.mean(y - np.matmul(X,b_rescaled))
        else:
            intercept = np.median(np.reshape(y - np.matmul(X,b_rescaled),(-1)))
        # This median calculation produces slightly different result in R and Py
        yfit = np.matmul(X,b_rescaled) + intercept   
        if (self.scaling!="None"):
            if (self.centring == "mean"):
                b0 = np.mean(yrc.Xs.astype("float64") - np.matmul(Xrc.Xs.astype("float64"),b))
            else:
                b0 = np.median(np.array(yrc.Xs.astype("float64") - np.matmul(Xrc.Xs.astype("float64"),b)))
            # yfit2 = (np.matmul(Xrc.Xs.astype("float64"),b) + b0)*yrc.col_sca + yrc.col_loc
            # already more generally included
        else:
            if (self.centring == "mean"):
                intercept = np.mean(y - np.matmul(X,b))
            else:
                intercept = np.median(np.array(y - np.matmul(X,b)))
            # yfit = np.matmul(X,b) + intercept
        yfit = yfit.reshape(-1)    
        r = y - yfit
        setattr(self,"x_weights_",W)
        setattr(self,"x_loadings_",P)
        setattr(self,"C_",res_snipls.C_)
        setattr(self,"x_scores_",T)
        setattr(self,"coef_",b_rescaled)
        setattr(self,"intercept_",intercept)
        setattr(self,"coef_scaled_",b)
        setattr(self,"intercept_scaled_",b0)
        setattr(self,"residuals_",r)
        setattr(self,"x_ev_",res_snipls.x_ev_)
        setattr(self,"y_ev_",res_snipls.y_ev_)
        setattr(self,"fitted_",yfit)
        setattr(self,"x_Rweights_",R)
        setattr(self,"x_caseweights_",wte)
        setattr(self,"y_caseweights_",wye)
        setattr(self,"caseweights_",we)
        setattr(self,"colret_",res_snipls.colret_)
        setattr(self,"x_loc_",Xrc.col_loc)
        setattr(self,"y_loc_",yrc.col_loc)
        setattr(self,"x_sca_",Xrc.col_sca)
        setattr(self,"y_sca_",yrc.col_sca)
        return(self)
        pass
    
        
    def predict(self,Xn):
        (n,p) = Xn.shape
        if p!= self.X.shape[1]:
            ValueError('New data must have seame number of columns as the ones the model has been trained with')
        return(np.matmul(Xn,self.coef_) + self.intercept_)
        
    def transform(self,Xn):
        (n,p) = Xn.shape
        if p!= self.X.shape[1]:
            ValueError('New data must have seame number of columns as the ones the model has been trained with')
        Xnc = robcent(Xn)
        Xnc.daprpr([self.x_loc_,self.x_sca_])
        return(Xnc.Xs*self.x_Rweights_)
        
    def weightnewx(self,Xn):
        (n,p) = Xn.shape
        if p!= self.X.shape[1]:
            ValueError('New data must have seame number of columns as the ones the model has been trained with')
        Tn = self.transform(Xn)
        scalet = self.scaling
        if (scalet == "None"):
            scalet = "mad"
        if isinstance(Tn,np.matrix):
            Tn = np.array(Tn)
        dtn = robcent(Tn)
        dtn.daprpr([self.centring, scalet])
        wtn = np.sqrt(np.array(np.sum(np.square(dtn.Xs),1),dtype=np.float64))
        wtn = wtn/np.median(wtn)
        wtn = wtn.reshape(-1)
        if (self.fun == "Fair"):
            wtn = self.Fair(wtn,self.probctx_)
        if (self.fun == "Huber"):
            wtn = self.Huber(wtn,self.probctx_)
        if (self.fun == "Hampel"):
            wtn = self.Hampel(wtn,self.probctx_,self.hampelbx_,self.hampelrx_)
        return(wtn)
        
    def valscore(self,Xn,yn,scoring):
        if scoring=='weighted':
            return(RegressorMixin.score(self,Xn,yn,sample_weight=self.caseweights_))
        elif scoring=='normal':
            return(RegressorMixin.score(self,Xn,yn))
        else:
            ValueError('Scoring flag must be set to "weighted" or "normal".')
        
    
            
        
        
    
