# sprm
Sparse partial robust M regression
==================================

Pyhton code for Sparse Partial Robust M regresion (SPRM)\[1\], a sparse and robust version of univariate partial least squares (PLS1). 

Description
-----------

The method performs four tasks at the same time in a single, consistent estimate: 
- regression: yields regression coefficients and predicts responses
- dimension reduction: calculates interpretable PLS-like components maximizing covariance to the predictand in a robust way 
- variable selection: depending on the paramter settings, can yield highly sparse regression coefficients that contain exact zero elements 
- outlier detection and compensation: yields a set of case weights in \[0,1\]. The lower the weight, the more outlying a case is. The estimate itself is outlier robust. 
        
The code is aligned to ScikitLearn, such that modules such as GridSearchCV can flawlessly be applied to it. 

The main SPRM implementation yields a class with the following structure:

Dependencies
------------
- From <sklearn.base>: BaseEstimator,TransformerMixin,RegressorMixin
- From <sklearn.utils>: _BaseComposition
- copy
- From <scipy.stats>: norm,chi2
- numpy 
- from <matplotlib>: pyplot. 

Parameters
----------
- eta: float. Sparsity parameter in \[0,1)
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
- snipls (class): sparse NIPALS regression (first described in: \[2\]) 
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
        runfile(".../robcent.py")
        runfile(".../prm.py")
        
- Estimate and predict by SPRM
        
        columnss = data.columns[2:8]
        res_sprm = sprm(2,.8,'Hampel',.95,.975,.999,'median','mad',True,100,.01,'ally','xonly',cols,True)
        res_sprm.fit(X0[:2666],y0[:2666])
        res_sprm.predict(X0[2666:])
        res_sprm.transform(X0[2666:])
        res_sprm.weightnewx(X0[2666:])
        res_sprm.get_params()
        res_sprm.set_params(fun="Huber")
        
- Cross-validated using GridSearchCV: 
        
        res_sprm_cv = GridSearchCV(sprm(), cv=10, param_grid={"n_components": \[1, 2, 3\], 
                                   "eta": np.arange(.1,.9,.05).tolist()})  
        res_sprm_cv.fit(X0[:2666],y0[:2666])  
        res_sprm_cv.best_params_
        
        
References
----------
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
        

Work to do
----------
- while the code is aligned with sklearn, it does not yet 100% follow the naming conventions therein
- optimize for speed 
- manipulations in robcent can be written more elegantly
- write custom plotting utilities
- suggestions always welcome
