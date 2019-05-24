#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:42:23 2019

Ancillary tools for plotting sprm results. 
    
    # Deleted: ABLine2D class, was broken in Py 3.7
    cv_score_table (function): tranform sklearn GridSearchCV 
        results into Data Frame

@author: Sven Serneels
"""
import numpy as np
import pandas as ps
        
def cv_score_table(res_sprm_cv):
        
    """
    Internal function reorganizing sklearn GridSearchCV results to pandas table. 
    The function adds the cv score table to the object as cv_score_table_
    """
        
    n_settings = len(res_sprm_cv.cv_results_['params'])
    etas = [res_sprm_cv.cv_results_['params'][i]['eta'] for i in range(0,n_settings)]
    components = [res_sprm_cv.cv_results_['params'][i]['n_components'] for i in range(0,n_settings)]
    cv_score_table_ = ps.DataFrame({'etas':etas, 'n_components':components, 'score':res_sprm_cv.cv_results_['mean_test_score']})
    return(cv_score_table_)