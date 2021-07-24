#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:29:12 2021

@author: goharshoukat
"""

import numpy as np
def logmovav(s1, s2, xdata, ydata):
    #Function for fft smoothing
    #variable sized window to average out the frequency. the size of the window
    #varies logarithmically. This is impportant as high frequency shows significant noise
    
    #inputs
    #s1: integer: starting window spanfor the moving average calculation. s1 
    #must be odd
    
    #s2: integer: ending window span for the moving average calc. s2 must be 
    #odd and s1 < s2
    
    #xdata : 1d array of real x values corresponding to the ydata to be filtered
    #ydata: 1D array of real data to be filtered
    
    #outputs
    #Fxdata: array of real subset of xdata matching the filtered Fydata
    #Fydata: array of real filtered data
    #Spans: array of integer: filtering of window spans
    
    n = len(ydata)
    
    #works out the end points of data so that the moving average algorithm can
    #operate without requiring data beyond what is available
    n1 = 1 + (s1-1)/2
    n2 = n - (s2-1)/2
    
    nn = n2 - n1 + 1 #number of elements after processed array
    
    #create the array of span values for the moving average computation
    Spans = np.logspace(np.log10(s1), np.log10(s2), int(nn))
    
    #convert all the elements of Spans to the nearest odd integer
    idx = Spans%2<1
    Spans = np.floor(Spans)
    Spans[idx] = Spans[idx] + 1
    
    dSpans = (Spans -1) / 2 #works out the 'half span -1' for each window span
    
    #carries out the moving average filtering
    Fydata = np.zeros(int(nn))
    for kk in range(0,int(nn)):
        ind= n1 + kk - 1
        Fydata[kk] = np.sum(ydata[int(ind - dSpans[kk]):int(ind + dSpans[kk])]/ Spans[kk])
        
    Fxdata = xdata[int(n1):int(n2)]
    
    return Fxdata, Fydata, Spans