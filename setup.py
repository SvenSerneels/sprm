#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:18:53 2018

@author: Sven serneels, Ponalytics
"""

from distutils.core import setup
import re
import sys
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sprm",
    version="0.0.1",
    author="Sven Serneels",
    author_email="svenserneels@gmail.com",
    description="Sparse Partial Robust M Regression, including plot functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SvenSerneels/sprm/",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
