#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:42:23 2019

Ancillary tools for plotting sprm results. 
    
    ABLine2D (class): first diagonal in plot
    cv_score_table (function): tranform sklearn GridSearchCV 
        results into Data Frame

@author: Sven Serneels
"""

import matplotlib.pyplot as pp 
import numpy as np
import pandas as ps

class ABLine2D(pp.Line2D):

    """
    Draw a line based on its slope and y-intercept. Additional arguments are
    passed to the <matplotlib.lines.Line2D> constructor.
    This class was grabbed from StackOverflow, not written by the main author
    """

    def __init__(self, slope, intercept, *args, **kwargs):

        # get current axes if user has not specified them
        if not 'axes' in kwargs:
            kwargs.update({'axes':pp.gca()})
        ax = kwargs['axes']

        # if unspecified, get the current line color from the axes
        if not ('color' in kwargs or 'c' in kwargs):
            kwargs.update({'color':ax._get_lines.color_cycle.next()})

        # init the line, add it to the axes
        super(ABLine2D, self).__init__([], [], *args, **kwargs)
        self._slope = slope
        self._intercept = intercept
        ax.add_line(self)

        # cache the renderer, draw the line for the first time
        ax.figure.canvas.draw()
        self._update_lim(None)

        # connect to axis callbacks
        self.axes.callbacks.connect('xlim_changed', self._update_lim)
        self.axes.callbacks.connect('ylim_changed', self._update_lim)

    def _update_lim(self, event):
        """ called whenever axis x/y limits change """
        x = np.array(self.axes.get_xbound())
        y = (self._slope * x) + self._intercept
        self.set_data(x, y)
        self.axes.draw_artist(self)
        
        
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