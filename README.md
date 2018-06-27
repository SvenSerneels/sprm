# sprm
Sparse partial robust M regression
==================================

Pyhton code for Sparse Partial Robust M regresion (SPRM), a sparse and robust version of univariate partial least squares (PLS1). 
        
The code is aligned to ScikitLearn, such that modules such as GridSearchCV can flawlessly be applied to it. 

To run a bare bones version, run
import pandas as ps
data = ps.read_csv("./Returns_shares.csv")
n_components = 2
eta = .8
pars = ["Hampel",0.95, 0.975, 0.999, "median", "mad", False, 100, .01,"ally","xonly"]
res_sprm = sprm(2,.8,pars,cols,True)
res_sprm.fit(X0[:2666],y0[:2666])
res_sprm.predict(X0[2667:])

Compared to R, this code offers a few more options regarding the staring values. The above yield identical results up to numerical precision. 

As by-products, this package also includes 
- a file for centering and scaling data robustly, robcent.py 
- a routine for sparse NIPALS regression, "snipls" (with mandatory centering)
The latter algorithm first outlined in: 
        
        
        
References
----------
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
        

Work to do
----------
- while the code is aligned with sklearn, it does not yet 100% follow the naming conventions therein
- optimize for speed 
- manipulations in robcent can be written more elegantly
- include rotations for new data on top of predictions. 
- write custom plotting utilities
