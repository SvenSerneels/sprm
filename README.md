# sprm
Sparse partial robust M regression
==================================

A scikit-learn compatible Python package for Sparse Partial Robust M regresion (SPRM)\[1\], a sparse and robust version of univariate partial least squares (PLS1). 


Description
-----------

The method performs four tasks at the same time in a single, consistent estimate: 
- *regression*: yields regression coefficients and predicts responses
- *dimension reduction*: calculates interpretable PLS-like components maximizing covariance to the predictand in a robust way 
- *variable selection*: depending on the paramter settings, can yield highly sparse regression coefficients that contain exact zero elements 
- *outlier detection and compensation*: yields a set of case weights in \[0,1\]. The lower the weight, the more outlying a case is. The estimate itself is outlier robust. 

Note: all the methods contained in this package have been designed for continuous data. They do not work correctly for caetgorical or textual data. 
        
The code is aligned to ScikitLearn, such that modules such as GridSearchCV can flawlessly be applied to it. 

The repository contains
- The estimator (sprm.py) 
- Plotting functionality based on Matplotlib (sprm_plot.py)
- Robust data pre-processing (robcent.py)
- The Sparse NIPALS (SNIPLS) estimator (snipls.py)
- Robust M regression estimator (rm.py)
- Ancillary functions for plotting (_plot_internals.py)
- Ancillary functions for M-estimation (_m_support_functions.py)

How to install
--------------
The package is distributed through PyPI, so install through: 
        
        pip install sprm 
    

The SPRM estimator
==================

The main SPRM implementation yields a class with the following structure:

Dependencies
------------
- From <sklearn.base>: BaseEstimator,TransformerMixin,RegressorMixin
- From <sklearn.utils>: _BaseComposition
- copy
- From <scipy.stats>: norm,chi2
- numpy 
- From <matplotlib>: pyplot. 
- From <statsmodels>: robust. 

Parameters
----------
- eta: float. Sparsity parameter in \[0,1). Note that eta=0 returns the non-sparse, yet robust, partial robust M-regression (PRM) \[2\]. 
- n_components: int > 1. Note that if applied on data, n_components shall take a value <= min(x_data.shape)
- fun: str, downweighting function. 'Hampel' (recommended), 'Fair' or 'Huber'
- probp1: float, probability cutoff for start of downweighting (e.g. 0.95)
- probp2: float, probability cutoff for start of steep downweighting (e.g. 0.975, only relevant if fun='Hampel')
- probp3: float, probability cutoff for start of outlier omission (e.g. 0.999, only relevant if fun='Hampel')
- centring: str, type of centring ('mean' or 'median', the latter recommended)
- scaling: str, type of scaling ('std','mad', the latter recommended, or 'None')
- verbose: boolean, specifying verbose mode
- maxit: int, maximal number of iterations in M algorithm
- tol: float, tolerance for convergence in M algorithm 
- start_cutoff_mode: str, value 'specific' will set starting value cutoffs specific to X and y (preferred); any other value will set X and y stating cutoffs identically. The non-specific setting yields identical results to the SPRM R implementation available from [CRAN](https://cran.r-project.org/web/packages/sprm/index.html).
- start_X_init: str, values 'pcapp' will include a PCA/broken stick projection to calculate the initial predictor block caseweights; any other value will just calculate initial predictor block case weights based on Euclidian distances within that block. The is less stable for very flat data (p >> n). 
- colums (def false): Either boolean or list. If False, no column names supplied. If a list (will only take length x_data.shape\[1\]), the column names of the x_data supplied in this list, will be printed in verbose mode
- copy (def True): boolean, whether to create deep copy of the data in the calculation process 

Attributes
----------
-  x_weights_: X block PLS weighting vectors (usually denoted W)
-  x_loadings_: X block PLS loading vectors (usually denoted P)
-  C_: vector of inner relationship between response and latent variablesblock re
-  x_scores_: X block PLS score vectors (usually denoted T)
-  coef_: vector of regression coefficients 
-  intercept_: intercept
-  coef_scaled_: vector of scaled regression coeeficients (when scaling option used)
-  intercept_scaled_: scaled intercept
-  residuals_: vector of regression residuals
-  x_ev_: X block explained variance per component
-  y_ev_: y block explained variance 
-  fitted_: fitted response
-  x_Rweights_: X block SIMPLS style weighting vectors (usually denoted R)
-  x_caseweights_: X block case weights
-  y_caseweights_: y block case weights
-  caseweights_: combined case weights
-  colret_: names of variables retained in the sparse model
-  x_loc_: X block location estimate 
-  y_loc_: y location estimate
-  x_sca_: X block scale estimate
-  y_sca_: y scale estimate

Methods
--------
- fit(X,y): fit model 
- predict(X): make predictions based on fit 
- transform(X): project X onto latent space 
- weightnewx(X): calculate X case weights
- getattr(): get list of attributes
- setattr(\*\*kwargs): set individual attribute of sprm object 
- valscore(X,y,scoring): option to use weighted scoring function in cross-validation if scoring=weighted 

Ancillary functions 
-------------------
- snipls (class): sparse NIPALS regression (first described in: \[3\]) 
- Hampel: Hampel weight function 
- Huber: Huber weight function 
- Fair: Fair weight function 
- brokenstick: broken stick rule to estimate number of relevant principal components  
- robcent (class): robust centring and scaling 

Example
-------
To run a toy example: 
- Source packages and data: 

        import pandas as ps
        data = ps.read_csv("./Returns_shares.csv")
        columns = data.columns[2:8]
        data = data.values[:,2:8]
        X = data[:,0:5]
        y = data[:,5]
        X0 = X.astype('float')
        y0 = y.astype('float')
        
- Estimate and predict by SPRM
        
        from sprm import sprm
        res_sprm = sprm(2,.8,'Hampel',.95,.975,.999,'median','mad',True,100,.01,'ally','xonly',columns,True)
        res_sprm.fit(X0[:2666],y0[:2666])
        res_sprm.predict(X0[2666:])
        res_sprm.transform(X0[2666:])
        res_sprm.weightnewx(X0[2666:])
        res_sprm.get_params()
        res_sprm.set_params(fun="Huber")
        
- Cross-validated using GridSearchCV: 
        
        import numpy as np
        from sklearn.model_selection import GridSearchCV 
        res_sprm_cv = GridSearchCV(sprm(), cv=10, param_grid={"n_components": [1, 2, 3], 
                                   "eta": np.arange(.1,.9,.05).tolist()})  
        res_sprm_cv.fit(X0[:2666],y0[:2666])  
        res_sprm_cv.best_params_
        
        
The Robust M (RM) estimator
===========================

RM has been implemented to be consistent with SPRM. It takes the same arguments, except for 'eta' and 'n_components', 
because it does not perform dimension reduction nor variable selection. For the same reasons, the outputs are limited to regression
outputs. Therefore, dimension reduction outputs like x_scores_, x_loadings_, etc. are not provided. 
        
  Estimate and predict by RM: 
  
        from sprm import rm
        res_rm = rm('Hampel',.95,.975,.999,'median','mad','specific',True,100,.01,columns,True)
        res_rm.fit(X0[:2666],y0[:2666])
        res_rm.predict(X0[2666:])
        
The Sparse NIPALS (SNIPLS) estimator
====================================

SNIPLS is the non-robust sparse univariate PLS algorithm \[3\]. SNIPLS has been implemented to be consistent with SPRM. It takes the same arguments, except for 'fun' and 'probp1' through 'probp3', since these are robustness parameters. For the same reasons, the outputs are limited to sparse dimension reduction and regression outputs. Robustness related outputs like x_caseweights_ cannot be provided.
        
  Estimate and predict by SNIPLS: 
  
        from sprm import snipls
        res_snipls = snipls(n_components=4, eta=.5)
        res_snipls.fit(X0[:2666],y0[:2666])
        res_snipls.predict(X0[2666:])
        
        
 
Plotting functionality
======================

The file sprm_plot.py contains a set of plot functions based on Matplotlib. The class sprm_plot contains plots for sprm objects, wheras the class sprm_plot_cv contains a plot for cross-validation. 

Dependencies
------------
- pandas
- numpy
- matplotlib.pyplot
- for plotting cross-validation results: sklearn.model_selection.GridSearchCV

Paramaters
----------
- res_sprm, sprm. An sprm class object that has been fit.  
- colors, list of str entries. Only mandatory input. Elements determine colors as: 
    - \[0\]: borders of pane 
    - \[1\]: plot background
    - \[2\]: marker fill
    - \[3\]: diagonal line 
    - \[4\]: marker contour, if different from fill
    - \[5\]: marker color for new cases, if applicable
    - \[6\]: marker color for harsh calibration outliers
    - \[7\]: marker color for harsh prediction outliers
- markers, a list of str entries. Elements determkine markers for: 
    - \[0\]: regular cases 
    - \[1\]: moderate outliers 
    - \[2\]: harsh outliers 
    
Methods
-------
- plot_coeffs(entity="coef_",truncation=0,columns=[],title=[]): Plot regression coefficients, loadings, etc. with the option only to plot the x% smallest and largets coefficients (truncation) 
- plot_yyp(ytruev=[],Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False): Plot y vs y predicted. 
- plot_projections(Xn=[],label=[],components = [0,1],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False): Plot score space. 
- plot_caseweights(Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False,mode='overall'): Plot caseweights, with the option to plot 'x', 'y' or 'overall' case weights for cases used to train the model. For new cases, only 'x' weights can be plotted. 

Remark
------
The latter 3 methods will work both for cases that the models has been trained with (no additional input) or new cases (requires Xn and in case of plot_ypp, ytruev), with the option to plot only the latter (option onlyval = True). All three functions have the option to plot case names if supplied as list.       

Ancillary classes
------------------ 
- sprm_plot_cv has method eta_ncomp_contour(title) to plot sklearn GridSearchCV results 
- ABline2D plots the first diagonal in y vs y predicted plots. 

Example (continued) 
-------------------
- initialize some values: 
        
        colors = ["white","#BBBBDD","#0000DD",'#1B75BC','#4D4D4F','orange','red','black']
        markers = ['o','d','v']
        label = ["AIG"]
        names = [str(i) for i in range(1,len(res_sprm.y)+1)]
        namesv = [str(i) for i in range(1,len(y0[2667:])+1)]
        
- run sprm.plot: 
        
        from sprm import sprm_plot
        res_sprm_plot = sprm_plot(res_sprm,colors)
        
- plot coefficients: 

        res_sprm_plot.plot_coeffs(title="All AIG SPRM scaled b")
        res_sprm_plot.plot_coeffs(truncation=.05,columns=columns,title="5% smallest and largest AIG sprm b")
        
  ![AIG sprm regression coefficients](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_b.png "AIG SPRM regression coefficients")

- plot y vs y predicted, training cases only: 

        res_sprm_plot.plot_yyp(label=label,title="AIG SPRM y vs. y predicted")
        res_sprm_plot.plot_yyp(label=label,names=names,title="AIG SPRM y vs. y predicted")

  ![AIG sprm y vs y predicted, taining set](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_yyp_train.png "AIG SPRM y vs y predicted, training set")
  
- plot y vs y predicted, including test cases
  
        res_sprm_plot.plot_yyp(ytruev=y0[2667:],Xn=X0[2667:],label=label,names=names,namesv=namesv,title="AIG SPRM y vs. 
                y predicted")            
        res_sprm_plot.plot_yyp(ytruev=y0[2667:],Xn=X0[2667:],label=label,title="AIG SPRM y vs. y predicted")
        
   ![AIG sprm y vs y predicted, taining set](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_yyp_train_test.png "AIG SPRM y vs y predicted")

- plot y vs y predicted, only test set cases: 

        res_sprm_plot.plot_yyp(ytruev=y0[2667:],Xn=X0[2667:],label=label,title="AIG SPRM y vs. y predicted",onlyval=True)
  
- plot score space, options as above, with the second one shown here: 

        res_sprm_plot.plot_projections(Xn=X0[2667:],label=label,names=names,namesv=namesv,title="AIG SPRM score space, components 1 and 2")
        res_sprm_plot.plot_projections(Xn=X0[2667:],label=label,title="AIG SPRM score space, components 1 and 2")
        res_sprm_plot.plot_projections(Xn=X0[2667:],label=label,namesv=namesv,title="AIG SPRM score space, components 1 and 2",onlyval=True)
        
  
   ![AIG sprm score space](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_T12.png "AIG SPRM score space")

- plot caseweights, options as above, with the second one shown here:

        res_sprm_plot.plot_caseweights(Xn=X0[2667:],label=label,names=names,namesv=namesv,title="AIG SPRM caseweights")
        res_sprm_plot.plot_caseweights(Xn=X0[2667:],label=label,title="AIG SPRM caseweights")
        res_sprm_plot.plot_caseweights(Xn=X0[2667:],label=label,namesv=namesv,title="AIG SPRM caseweights",onlyval=True)  
        
   ![AIG sprm caseweights](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_caseweights.png "AIG SPRM caseweights")
   

- plot cross-validation results: 

        from sprm import sprm_plot_cv
        res_sprm_plot_cv = sprm_plot_cv(res_sprm_cv,colors)
        res_sprm_plot_cv.eta_ncomp_contour()
        res_sprm_plot_cv.cv_score_table_
        
  ![AIG sprm CV results](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_CV.png "AIG SPRM CV results")
  
        
References
----------
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Partial robust M regression](https://doi.org/10.1016/j.chemolab.2005.04.007), Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
        

Release notes
=============

Version 0.2.1
-------------
- sprm now takes both numeric (n,1) np matrices and (n,) np.arrays as input 


Version 0.2.0
-------------
Changes compared to version 0.1: 
- All functionalities can now be loaded in modular way, e.g. to use plotting functions, now source the plot function separately:
        
        from sprm import sprm_plot 
        
- The package now includes a robust M regression estimator (rm.py), which is a multiple regression only variant of sprm. 
  It is based on the same iterative re-weighting scheme, buit does not perform dimension reduction, nor variable selection.
- The robust preprocessing routine (robcent.py) has been re-written so as to be more consistent with sklearn.

Version 0.3
-----------
All three estimators provided as separate classes in module:

        from sprm import sprm 
        from sprm import snipls
        from sprm import rm
        
Plus some minor changes to make it consistent with the latest numpy and matplotlib versions. 

Work to do
----------
- optimize alignment to sklearn
- optimize for speed 
- extend to multivariate responses (open research topic !)
- suggestions always welcome 