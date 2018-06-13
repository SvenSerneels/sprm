#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:56:12 2018

Class containing:
    Sparse Partial Robust M Regression (SPRM)
    Sparse NIPALS (SNIPLS) 

Depends on robcent class for robustly centering and scaling data 

@author: Sven Serneels, Ponalytics
"""
import numpy as np
import matplotlib.pyplot as pp
from scipy.stats import norm, chi2
from sklearn.base import RegressorMixin,BaseEstimator,TransformerMixin
class MyException(Exception):
        pass

class snipls(BaseEstimator,TransformerMixin,RegressorMixin):
    
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



class sprm(BaseEstimator,TransformerMixin,RegressorMixin,robcent):
    
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
    pars: list containing minor tuning parameters: 
        pars[0]: str, downweighting function. 'Hampel' (recommended), 'Fair' or 
                 'Huber'
        pars[1]: float, probability cutoff for start of downweighting 
                 (e.g. 0.95)
        pars[2]: float, probability cutoff for start of steep downweighting 
                 (e.g. 0.975)
        pars[3]: float, probability cutoff for start of outlier omission 
                 (e.g. 0.999)
        pars[4]: str, type of centering ('mean' or 'median' [recommended])
        pars[5]: str, type of scaling ('std','mad' [recommended] or 'None')
        pars[6]: boolean, specifying verbose mode
        pars[7]: int, =max_iter, maximal number of iterations in M algorithm
        pars[8]: float, =tol, tolerance for convergence in M algorithm 
        pars[9]: str, 
                 'specific' will set starting vlue cutoffs X and y specific, 
                 any other value will set X and y stating cutoffs identically
        pars[10]: str, 'pcapp' will include a PCA/ broken stick projection to 
                  calculate the staring weights, else just based on X
        
    colums (def false): Either boolean or list
        if False, no column names supplied 
        if a list (will only take length x_data.shape[1]), the column names of 
            the x_data supplied in this list, will be printed in verbose mode
    copy (def True): boolean, whether to copy data
        Note: copy not yet aligned with sklearn def  - we always copy
    
    """
    
    def __init__(self,n_components=1,eta=.5,pars=['Hampel', 0.95, 0.975, 0.999, 'median', 'mad', True, 100, 0.01, 'specific','pcapp']
                ,columns=False,copy=True):
        self.n_components = int(n_components) 
        self.eta = float(eta)
        self.pars = pars
        self.columns = columns
        self.copy = copy
        
    def brokenstick(self,n_components):
        q = np.triu(np.ones((n_components,n_components)))
        r = np.empty((n_components,1),float)
        r[0:n_components,0] = (range(1,n_components+1))
        q = np.matmul(q,1/r)
        q /= n_components
        return q

     #  pars = ["Hampel",0.95, 0.975, 0.999, "median", "mad", False, 100, .01 ]
    def fit(self,X,y):
        if self.copy:
            self.X = X
            self.y = y
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
        if (not(self.pars[0] in ("Hampel", "Huber", "Fair"))):
            raise MyException("Invalid weighting function. Choose Hampel, Huber or Fair for parameter fun.")
        if ((self.pars[1] > 1) | (self.pars[1] <= 0)):
            raise MyException("probp1 is a probability. Choose a value between 0 and 1")
        if (self.pars[0] == "Hampel"):
            if (not((self.pars[1] < self.pars[2]) & (self.pars[2] < self.pars[3]) & (self.pars[3] <= 1))):
                raise MyException("Wrong choise of parameters for Hampel function. Use 0<probp1<hampelp2<hampelp3<=1")
        ny = y.shape[0]
        if ny != n:
            raise MyException("Number of cases in y and X must be identical.")
        Xrc = robcent(X)
        yrc = robcent(y)
        Xrc.daprpr(self.pars[4:6])
        yrc.daprpr(self.pars[4:6])
        #datac <- attr(datamc, "Center")
        #datas <- attr(datamc, "Scale")
        #attr(datac, "Type") <- center
        y0 = y
        ys = yrc.Xs.astype('float64')
        Xs = Xrc.Xs.astype('float64')
        if (self.pars[10]=='pcapp'):
            U, S, V = np.linalg.svd(Xs)
            spc = np.square(S)
            spc /= np.sum(spc)
            relcomp = max(np.where(spc - self.brokenstick(min(p,n))[:,0] <=0)[0][0],1)
            Urc = robcent(np.array(U[:,0:relcomp]))
            Urc.daprpr(self.pars[4:6])
        else: 
            Urc = Xrc
        wx = np.sqrt(np.array(np.sum(np.square(Urc.Xs),1),dtype=np.float64))
        wx = wx/np.median(wx)
        if self.pars[4:6]==['median','mad']:
            wy = np.array(abs(yrc.Xs),dtype=np.float64)
        else:
            wy = (y - np.median(y))/(1.4826*np.median(abs(y-np.median(y))))
        probcty = norm.ppf(self.pars[1])
        if self.pars[9] == 'specific':
            probctx = chi2.ppf(self.pars[1],relcomp)
        else: 
            probctx = probcty
        if (self.pars[0] == "Fair"):
            wx = 1/(1 + abs(wx/(probctx * 2)))**2
            wy = 1/(1 + abs(wy/(probcty * 2)))**2
        if (self.pars[0] == "Huber"):
            wx[np.where(wx <= probctx)] = 1
            wx[np.where(wx > probctx)] = probctx/abs(wx[np.where(wx > probctx)])
            wy[np.where(wy <= probcty)] = 1
            wy[np.where(wy > probcty)] <- probcty/abs(wy[np.where(wy > probcty)])
        if (self.pars[0] == "Hampel"):
            hampelby = norm.ppf(self.pars[2])
            hampelry = norm.ppf(self.pars[3])
            if self.pars[9] == 'specific':
                hampelbx = chi2.ppf(self.pars[2],relcomp)
                hampelrx = chi2.ppf(self.pars[3],relcomp)
            else: 
                hampelbx = hampelby
                hampelrx = hampelry
            wx[np.where(wx <= probctx)[0]] = 1
            wx[np.where((wx > probctx) & (wx <= hampelbx))[0]] = probctx/abs(wx[
                    np.where((wx > probctx) & (wx <= hampelbx))[0]])
            wx[np.where((wx > hampelbx) & (wx <= hampelrx))[0]] = np.divide(
                        probctx * (hampelrx - (wx[np.where((wx > hampelbx) 
                        & (wx <= hampelrx))[0]])),
                       (hampelrx - hampelbx) * abs(wx[np.where((wx > hampelbx) &
                       (wx <= hampelrx))[0]])
                        )
            wx[np.where(wx > hampelrx)[0]] = 0
            wy[np.where(wy <= probcty)[0]] = 1
            wy[np.where((wy > probcty) & (wy <= hampelby))[0]] = np.divide(probcty,
               abs(wy[np.where((wy > probcty) & (wy <= hampelby))[0]]))
            wy[np.where((wy > hampelby) & (wy <= hampelry))[0]] = np.divide(probcty * 
               (hampelry - (wy[np.where((wy > hampelby) & (wy <= hampelry))[0]])),
               (hampelry-hampelby) * abs(wy[np.where((wy > hampelby) & (wy <= hampelry))[0]]))
            wy[np.where(wy > hampelry)[0]] = 0
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
        while ((difference > self.pars[8]) & (loops < self.pars[7])):
            res_snipls = snipls(self.eta,self.n_components,
                            self.pars[6],self.columns,self.copy)
            res_snipls.fit(Xw,yw)
            T = np.divide(res_snipls.x_scores_,WEmat[:,0:(self.n_components)])
            b = res_snipls.coef_
            yp = res_snipls.fitted_
            r = ys - yp
            if (len(r)/2 > np.sum(r == 0)):
                r = abs(r)/(1.4826 * np.median(abs(r)))
            else:
                r = abs(r)/(1.4826 * np.median(abs(r[r != 0])))
            scalet = self.pars[5]
            if (scalet == "None"):
                scalet = "mad"
            dt = robcent(T)
            dt.daprpr([self.pars[4], scalet])
            wtn = np.sqrt(np.array(np.sum(np.square(dt.Xs),1),dtype=np.float64))
            wtn = wtn/np.median(wtn)
            wtn = wtn.reshape(-1)
            wye = r
            wte = wtn
            if (self.pars[0] == "Fair"):
                wte = 1/(1 + abs(wte/(probctx * 2)))**2
                wye = 1/(1 + abs(wye/(probcty * 2)))**2
            if (self.pars[0] == "Huber"):
                wte[np.where(wtn <= probctx)] = 1
                wte[np.where(wtn > probctx)] = probctx/abs(wtn[np.where(wtn > probctx)])
                wye[np.where(wye <= probcty)] = 1
                wye[np.where(wye > probcty)] <- probcty/abs(wye[np.where(wye > probcty)])
            if (self.pars[0] == "Hampel"):
                probctx = chi2.ppf(self.pars[1],self.n_components)
                hampelbx = chi2.ppf(self.pars[2],self.n_components)
                hampelrx = chi2.ppf(self.pars[3],self.n_components)
                wte[np.where(wtn <= probctx)[0]] = 1
                wte[np.where((wtn > probctx) & (wtn <= hampelbx))[0]] = probctx/abs(wtn[
                    np.where((wtn > probctx) & (wtn <= hampelbx))[0]])
                wte[np.where((wtn > hampelbx) & (wtn <= hampelrx))[0]] = np.divide(
                        probctx * (hampelrx - (wtn[np.where((wtn > hampelbx) 
                        & (wtn <= hampelrx))[0]])),
                       (hampelrx - hampelbx) * abs(wtn[np.where((wtn > hampelbx) &
                       (wtn <= hampelrx))[0]])
                        )
                wte[np.where(wtn > hampelrx)[0]] = 0
                wye[np.where(wye <= probcty)[0]] = 1
                wye[np.where((wye > probcty) & (wye <= hampelby))[0]] = np.divide(probcty,
                    abs(wye[np.where((wye > probcty) & (wye <= hampelby))[0]]))
                wye[np.where((wye > hampelby) & (wye <= hampelry))[0]] = np.divide(probcty * 
                    (hampelry - (wye[np.where((wye > hampelby) & (wye <= hampelry))[0]])),
                    (hampelry-hampelby) * abs(wye[np.where((wye > hampelby) & (wye <= hampelry))[0]]))
                wye[np.where(wye > hampelry)[0]] = 0
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
        if (difference > self.pars[7]):
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
        Xrw.daprpr([self.pars[4],"None"]) 
        # these four lines likely phantom code 
        # in R implementation multiplication just with centered data 
                
        # T = Xrw.Xm.astype("float64")*R 
        T = Xs * R
        if self.pars[6]:
            print("Final Model: Variables retained for " + str(n_components) + " latent variables: \n" 
                 + str(res_snipls.colret_) + "\n")
        b_rescaled = np.reshape(yrc.col_sca/Xrc.col_sca,(5,1))*b
        if(pars[4] == "mean"):
            intercept = np.mean(y - np.matmul(X,b_rescaled))
        else:
            intercept = np.median(np.reshape(y - np.matmul(X,b_rescaled),(-1)))
        # This median calculation produces slightly different result in R and Py
        yfit = np.matmul(X,b_rescaled) + intercept   
        if (self.pars[5]!="None"):
            if (self.pars[4] == "mean"):
                b0 = np.mean(yrc.Xs.astype("float64") - np.matmul(Xrc.Xs.astype("float64"),b))
            else:
                b0 = np.median(np.array(yrc.Xs.astype("float64") - np.matmul(Xrc.Xs.astype("float64"),b)))
            # yfit2 = (np.matmul(Xrc.Xs.astype("float64"),b) + b0)*yrc.col_sca + yrc.col_loc
            # already more generally included
        else:
            if (self.pars[4] == "mean"):
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
        return(np.matmul(Xn,self.coef_) + self.intercept_)
        
    def valscore(self,Xn,yn,scoring):
        if scoring=='weighted':
            return(RegressorMixin.score(self,Xn,yn,sample_weight=self.caseweights_))
        elif scoring=='normal':
            return(RegressorMixin.score(self,Xn,yn))
        else:
            ValueError('Scoring flag must be set to "weighted" or "normal".')
        
    
            
        
        
    
