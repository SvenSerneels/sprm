#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:17:17 2018

@author: Sven Serneels, Ponalytics
"""

__name__ = "sprm"
__author__ = "Sven Serneels"
__license__ = "MIT"
__version__ = "0.5.0"
__date__ = "2020-03-28"

from .preprocessing.robcent import VersatileScaler
from .sprm.sprm import sprm
from .sprm.snipls import snipls
from .sprm.rm import rm
from .sprm.sprm_plot import sprm_plot,sprm_plot_cv




