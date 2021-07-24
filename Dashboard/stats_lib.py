#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:02:01 2021

@author: goharshoukat
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy import fftpack

def fft(signal: np.ndarray, sampling_frequency, bins):
    #FFT function calculates the fft for the signal
    #if normalisation is needed, the normalise argument should be True
    #and omega which is the rotational frequency needs to be given
    #bins gives the size of the fft output. 
    #if normalisation is required, omega must also be provided. 
    
    no_of_meas = bins
    freq = np.linspace(0, int(sampling_frequency/2), int(no_of_meas/2))
    freq_data = fftpack.fft(signal, bins)
    y = 2/no_of_meas * np.abs(freq_data[0:np.int(no_of_meas/2)])
    y[0] = y[0] / 2
    
    return y

def fft_plot(spec, column, frequency, rotor_freq, units, path):
    #This function rejects the 0th DC value as the log function fails.
    #Function allows to plot and save normalised frequency ffts
    bins = len(spec)
    normalised_frequency = frequency/rotor_freq
    fig, (f1, f2) = plt.subplots(2,1)
    f1.loglog(frequency[1:bins], spec[1:])
    f2.loglog(normalised_frequency[1:bins],spec[1:])
    fig.suptitle(column)
    f2.set_xlabel(r'$\frac{F}{f_0}$ [Hz/Hz]')
    f1.set_xlabel('Frequency [Hz]')
    f1.set_ylabel( 'Amplitude' '(' + units[column]+ ')')
    f2.set_ylabel( 'Amplitude' '(' + units[column]+ ')')
    f1.set_title('Non-Normalised')
    f2.set_title('Normalised')
    f1.grid()
    f2.grid() 
    fig.tight_layout()
    plt.savefig(path + column + '.png')
    plt.close()

        
    
    
    
def rolling_average(df, column_name: str, value_skip, normalization=False):
    if not isinstance(column_name, str):
        raise TypeError
    if not isinstance(normalization, bool):
        raise TypeError
    arr = df[column_name].to_numpy()
    stat = []
    stat = np.append(stat, arr[0])
    
    for i in range(1,len(arr), value_skip):
        var = (arr[i] + stat[int(i/value_skip) - 1] * (len(stat)))/(len(stat)+1)
        stat = np.append(stat, var)
    
    if normalization == True:
        mean = np.mean(arr)
        stat_norm = stat/mean
        return stat_norm, stat
    else:
        return stat

def rolling_std(df, column_name, value_skip, normalization= False):
    if not isinstance(column_name, str):
        raise TypeError
    if not isinstance(normalization, bool):
        raise TypeError
    
    arr = df[column_name].to_numpy()
    #the dummy array holds the sampled array
    dummy = []
    
    for i in range(0, len(arr), value_skip):
        dummy = np.append(dummy, arr[i])
    
    stat = []
    
    for i in range(0, len(dummy)):
        var = np.std(dummy[0:(i+1)])
        stat = np.append(stat, var)
    
    if normalization==True:
        mean = np.mean(arr)
        stat_norm = stat/mean
        return stat_norm, stat
    else:
        return stat
        
def rolling_max(df, column_name):
    if not isinstance(column_name, str):
        raise TypeError
    arr = df[column_name].to_numpy()
    stat = []   
    stat = np.append(stat, arr[0])

    for i in range(1, len(arr)):
        if stat[i-1] > arr[i]:
            stat = np.append(stat, stat[i-1])
        else:
            stat = np.append(stat, arr[i])
    return stat
def rolling_min(df, column_name):    
    
    if not isinstance(column_name, str):
        raise TypeError
    arr = df[column_name].to_numpy()
    stat = []   
    stat = np.append(stat, arr[0])

    for i in range(1, len(arr)):
        if stat[i-1] < arr[i]:
            stat = np.append(stat, stat[i-1])
        else:
            stat = np.append(stat, arr[i])
    return stat
   
    

        
def plot_rolling_props(df, units, path, prop, df_norm = pd.DataFrame()):
    #Function to plot and save the mean of each data stream
    #if df_norm provided, then subfigures created. 
    #otherwise figures will be created
    #if the normalised data frame is passed, it should use the keyword for this
    #function df_norm

    
    if df_norm.empty:        
        if prop == 'mean':        
            for column in df.columns:        
                plt.plot(df.index, df[column])
                plt.title(column)
                plt.xlabel('time (s)')
                plt.ylabel(r'$\mu$' + '(' + units[column]+ ')')
                plt.savefig(path + column + '.png')
                plt.grid()
                plt.close()
        
        elif prop == 'max' or 'min':
            for column in df.columns:        
                plt.plot(df.index, df[column])
                plt.title(column)
                plt.xlabel('time (s)')
                plt.ylabel(units[column])
                plt.grid()
                plt.savefig(path + column + '.png')
                plt.close()
        
        else:
            for column in df.columns:        
                plt.plot(df.index, df[column])
                plt.title(column)
                plt.xlabel('time (s)')
                plt.ylabel(r'$\sigma$' + '(' + units[column]+ ')')
                plt.savefig(path + column + '.png')
                plt.grid()
                plt.close()

    else:
        if prop == 'mean':     
            for column in df.columns:        
                fig, (f1, f2) = plt.subplots(2,1, sharex = True)
                f1.plot(df[column])
                f2.plot(df_norm[column])
                fig.suptitle(column)
                f2.set_xlabel('time (s)')
                f1.set_ylabel(r'$\mu$' + '(' + units[column]+ ')')
                f2.set_ylabel(r'$\frac{\mu}{\mu}$')
                f1.set_title('Non-Normalised')
                f2.set_title('Normalised')
                f1.grid()
                f2.grid()
                fig.savefig(path + column + '.png')
                plt.close()
                
        else:
            for column in df.columns:
                fig, (f1, f2) = plt.subplots(2,1,sharex = True)
                fig.suptitle(column)
                f1.plot(df[column])
                f2.plot(df_norm[column])
                f2.set_xlabel('Time [s]')
                f1.set_title('Non-Normalised')
                f2.set_title('Normalised')
                f1.grid()
                f2.grid()
                f1.set_ylabel(r'$\sigma$' + '(' + units[column]+ ')')
                f2.set_ylabel(r'$\frac{\sigma}{\mu}$')
                fig.savefig(path + column + '.png')
                plt.close()
            