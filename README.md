# sprm - Sparse Partial Robust M Regression 
Python code for Sparse Partial Robust M Regresion, a sparse and robust version of partial least squares.

The algorithm was first outlined in: 
        _Sparse partial robust M regression, 
        Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, 
        Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59._
        
The code is aligned to `scikit-learn`, such that modules such as `GridSearchCV` can flawlessly be applied to it.

To run a bare bones version, run
```python
import pandas as ps

data = ps.read_csv("./Returns_shares.csv")
n_components = 2
eta = 0.8
pars = ["Hampel", 0.95, 0.975, 0.999, "median", "mad", False, 100, .01, "ally", "xonly"]
res_sprm = sprm(n_components, eta, pars, False, True)
res_sprm.fit(X0[:2666], y0[:2666])
res_sprm.predict(X0[2667:])
```

Compared to R, this code offers a few more options regarding the staring values. The above yield identical results up to numerical precision.

As by-products, this package also includes 
- a file for centering and scaling data robustly, `robcent.py`
- a routine for sparse NIPALS regression, `snipls` (with mandatory centering)
The latter algorithm was first outlined in: 
        _Sparse and robust PLS for binary classification, 
        I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, 
        Journal of Chemometrics, 30 (2016), 153-162._
        

Work to do: 
- while the code is aligned with `scikit-learn`, it does not yet 100% follow the naming conventions therein
- optimize for speed
- manipulations in `robcent` can be written more elegantly
- include rotations for new data on top of predictions
- write custom plotting utilities
