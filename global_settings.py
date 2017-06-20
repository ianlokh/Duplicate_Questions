#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:12:06 2017

@author: ianlo
"""

# import importlib
# importlib.reload(gs)

def init():
    global var_threshold
    global seedvalue
    global word2vecmodel
    global chunksize
    global maxqnstrlen
    
    # default values
    word2vecmodel = ''
    var_threshold = 0.01
    seedvalue = 37458
    chunksize = 50000
    maxqnstrlen = 1141
