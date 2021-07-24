#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:42:23 2021

@author: goharshoukat
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import math
import os
import pickle
import stats_lib
from csaps import csaps
try: from Dashboard.logmovav import logmovav
except: from logmovav import logmovav
from sklearn.preprocessing import scale
import cmath
#%%
'''
File Read into script
Data Cleaning, Time Stamp Addition
Simple Statistics Calculated
Initial Plots
'''
def access_file(file):
    df = pd.read_csv(file, sep = '\t', skiprows=9, low_memory=False)
    units = df.iloc[0]
    df = df.drop([0])
    #renaming first column header
    try: df = df.rename({'#rpm': 'RPM'}, axis=1)
    except: df = df.rename({'#RPM': 'RPM'}, axis=1)

    #removing the hash sign and renaming the dataframe index, important for plotting grapsh
    try: units['#rpm'] = 'RPM'
    except: units['#RPM'] = 'RPM'

    try: units = units.rename({'#rpm': 'RPM'})
    except: units = units.rename({'#RPM': 'RPM'})
    #all column heads are stored to recreate proper columns
    col_head = list(df.columns)
    #the data series is throwing an exception that the data is not numeric
    #the following for loop converts the entire dataframe to numeric which
    #can then be plotted
    for i in range(len(df.columns)):
        x = pd.to_numeric(df[col_head[i]])
        df = df.drop(columns=[col_head[i]])
        df.insert(i, col_head[i], x)
        del x

    sampling_freq = read_freq(file)
    no_of_meas = len(df)
       # create time arra
    t = np.linspace(0, no_of_meas/sampling_freq, no_of_meas)
    #first add column to the data framce
    df['time'] = t
    #rename index to time values
    df = df.set_index('time')
    #the following try except statements accomodate for the different column headers
    try: df['Force'] = -df['Force']
    except: df['Thrust'] = -df['Thrust']
    #df['Fx'] = -df['Fx']
    try: df['Couple'] = -df['Couple']
    except: df['Torque'] = -df['Torque']

    return df, units

def read_freq(file):
    #code to read frequency from file
    run_parameters = []
    a_file = open(file)
    lines_to_read = np.arange(0, 8, 1)

    for position, line in enumerate(a_file):
        if position in lines_to_read:
            run_parameters.append(line)

    return (int(run_parameters[3][-9:-6]))

def stats(ser):
    #Statistics of the data
    std_val = ser.std(axis = 0)                    # standard deviation
    mean_val = ser.mean(axis = 0)                  # arithmatic mean
    max_val = ser.max(axis = 0)                    # max values
    min_val = ser.min(axis = 0)                    # min values
    return std_val, mean_val, max_val, min_val

def spectral_analysis(df, column, bins = False):

    if bins == 'Default':

        bins = len(df)
        timestep = df.index[1]
    else:
        timestep = df.index[1]

    sampling_freq = 1/timestep
    spec = stats_lib.fft(df[column].to_numpy(), int(sampling_freq), int(bins))
    frequency = np.fft.fftfreq(int(bins), d=timestep)

    #call on the smoothing function. pass on the spectral analysis values
    s1 = (5) #define the first moving average window. these are hard coded into the program. future versions of the dashboard, we can adjust this.
    s2 = (69) #define the second moving average window.
    Fxdata, Fydata, Spans = logmovav(s1, s2, frequency, spec)

    #return spec, frequency
    return Fydata, Fxdata

#noise spectral analysis
def spectral_analysis_noisy(df, column, bins = False):

    if bins == 'Default':

        bins = len(df)
        timestep = df.index[1]
    else:
        timestep = df.index[1]
    sampling_freq = 1/timestep
    spec = stats_lib.fft(df[column].to_numpy(), int(sampling_freq), int(bins))
    frequency = np.fft.fftfreq(int(bins), d=timestep)

    return spec, frequency
#function to cacluate rotor frequency to normalise fft
def rotor_freq(ser):
    return ser.mean()/60

# %% Filtrage Function
# x is the signal input after subtracting the mean from the input
# fc is an array of high and low frequency between which the signal filteration will be done
# filter type can be ignore here
def filtrage(t,x,fc,filter_type):
    fe = 1/(t[1]-t[0])
    n = len(x)
    #not clear on why n-1 is used
    delta_f = fe/(n-1)
    #np.arrange does not work well with non-integer values and hence the error in this funciton with the mismatch in length.
    #replacing this with linspace resolves the error
    #f = np.arange(-fe/2, fe/2+delta_f, delta_f)
    #low pass filter
    f = np.linspace(-fe/2, fe/2, n)
    f1 = min(fc)
    f2 = max(fc)

    ind1 = abs(f) < f1
    ind2 = abs(f) > f2
    ind = (ind1 + ind2)

    ind = np.fft.fftshift(ind)
    fftx = np.fft.fft(x)

    fftx[ind == True] = 0
    xf = np.real(np.fft.ifft(fftx))
    return xf
# %% Anglue Turbine

def angle_turbine(t, Fy1, Fy2, Fy3, fr):

    f1 = fr - 0.2
    f2 = fr + 0.2


    Fy1=filtrage(t,Fy1-np.mean(Fy1),[f1, f2],3)
    Fy2=filtrage(t,Fy2-np.mean(Fy2),[f1, f2],3)
    Fy3=filtrage(t,Fy3-np.mean(Fy3),[f1, f2],3)

    theta1=np.unwrap(np.arctan2((+np.sqrt(3)*Fy1),(Fy1+2*Fy2)))
    theta2=np.unwrap(np.arctan2(-(-np.sqrt(3)*Fy1),-(Fy1+2*Fy3)))
    theta3=np.unwrap(np.arctan2((-np.sqrt(3)*(Fy2+Fy3)),(Fy2-Fy3)))
    theta = np.mean(np.column_stack([theta1, theta2, theta3]), axis = 1)
    theta=theta-np.round(theta/(2*np.pi))*(2*np.pi)
    theta1=theta1-np.round(theta1/(2*np.pi))*(2*np.pi)
    theta2=theta2-np.round(theta2/(2*np.pi))*(2*np.pi)
    theta3=theta3-np.round(theta3/(2*np.pi))*(2*np.pi)
    df = pd.DataFrame({'Fy1': Fy1, 'Fy2':Fy2, 'Fy3':Fy3, 'theta1':theta1, 'theta2':theta2, 'theta3':theta3, 'theta':theta})
    df.index = t
    return df

#content is reference to the signal seletion made by the user.
#since the entire df is passed, one particular signal type needs to be chosen
def angle_theta(df, content):
    t = df.index
    Fy1 = df['Fy1'].to_numpy()
    Fy2 = df['Fy2'].to_numpy()
    Fy3 = df['Fy3'].to_numpy()
    fr = rotor_freq(df['RPM'])
    angle = angle_turbine(t, Fy1, Fy2, Fy3, fr)
    #plt.scatter(angle_theta['theta'], df[content], s= 0.01)
    newdf = pd.DataFrame({'theta':angle['theta'], 'signal': df[content]})

    angle_df_sorted = newdf.sort_values(by = 'theta')
    theta_s = np.arange(-np.pi, np.pi, np.pi/len(newdf))
    fx_s = csaps(angle_df_sorted['theta'], angle_df_sorted['signal'], theta_s, smooth = .9)
    #plt.plot(theta_s, fx_s)
    fit_df = pd.DataFrame({'theta_s': theta_s, 'Average': fx_s})
    return angle_df_sorted, fit_df

def angle_theta_binning(df, content, bins = 360):
    t = df.index
    Fy1 = df['Fy1'].to_numpy()
    Fy2 = df['Fy2'].to_numpy()
    Fy3 = df['Fy3'].to_numpy()

    try: fr = rotor_freq(df['RPM'])
    except: fr = rotor_freq(df['#RPM'])
    angle = angle_turbine(t, Fy1, Fy2, Fy3, fr)

    newdf = pd.DataFrame({'theta':(angle['theta']), 'signal': df[content]})
    angle_df_sorted = newdf.sort_values(by = 'theta')
    binned_df = pd.cut(angle_df_sorted.theta, bins)
    averaged_binned_df = angle_df_sorted.groupby(binned_df).mean()

    return angle_df_sorted, averaged_binned_df



def polar_chart(df, content, normalise = False):
    angle, fit= angle_theta(df, content)
    if normalise == True:
        scale(angle['signal'], with_mean=True, with_std= True, copy = False)
        scale(fit['Average'], with_mean=True, with_std=True, copy = False)


    fig = plt.figure(figsize = (60,60))
    ax = fig.add_subplot(111,position=[0.10, 0.10, 0.70, 0.70], projection='polar')
    #plt.autoscale(axis = 'both', tight = 'bool')
            #pass values to the function angle_turbine
    ax.set_rlabel_position(90)
    ax.set_theta_zero_location("N")  # theta=0 at the top
    ax.set_theta_direction(-1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.scatter(angle['theta'], angle['signal'], s=0.01)
    #ax.scatter(angle_binned['theta'], angle_binned['signal'], s=0.01)
    ax.plot(fit['theta_s'], fit['Average'], color = 'red')
    #ax.plot(np.deg2rad(average_binned['theta']), average_binned['signal'], color = 'red')
            #self.ax4.set_ylim(min(angle['signal'])-1, max(angle['signal']+0.5))
            #half the samples are plotted to avoid clutter


            #self.ax4.scatter(self.theta, self.y_new, s = 0.5, alpha = 1)
    ax.set_title('Varation against Blade Angle')

#this function uses the angle_theta_binning
def polar_chart_binning(df, content, normalise = False):
    angle, average_binned= angle_theta_binning(df, content)
    if normalise == True:
        scale(angle['signal'], with_mean=True, with_std= True, copy = False)
        scale(average_binned['signal'], with_mean=True, with_std=True, copy = False)

    fig2 =plt.figure(figsize = (60,60))
    ax1 = fig2.add_subplot(111,position=[0.10, 0.10, 0.70, 0.70], projection='polar')
    ax1.set_rlabel_position(90)
    ax1.set_theta_zero_location("N")  # theta=0 at the top
    ax1.set_theta_direction(-1)
    #ax1.set_ylabel(content + '(' + units[content] +')', rotation=)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.scatter(angle['theta'], angle['signal'], s=0.01)
    ax1.plot((average_binned['theta']), average_binned['signal'], color = 'red')
    ax1.set_title('Varation against Blade Angle')




# %% Calculation of Mean at each degree

def fitting(signal_, theta, plot = False):

    theta_s = np.arange(-np.pi, np.pi, np.pi/len(signal_))
    newdf= pd.DataFrame({'theta': theta, 'signal': signal_})
    newdf = newdf.sort_values(by = 'theta')
    avg = csaps(theta, signal, theta_s, smooth = .95)

    if plot == True:
        plt.scatter(np.rad2deg(newdf['theta']), newdf['signal'], s =0.01)
        plt.plot(np.rad2deg(theta_s), avg)
        plt.xlabel('time[s]')
        #plt.ylabel()
    return avg, theta_s

# %% set limits for the polar plot using averaging and expressing all points as a percentage of mean

def limits_polar_plot(series):
    mean = np.mean(series)
    return (series-mean)/mean*100

def phase_shift(signal, shift):
    signalFFT = np.fft.rfft(signal)

    ## Get Power Spectral Density
    signalPSD = np.abs(signalFFT) ** 2
    signalPSD /= len(signalFFT)**2



    ## Phase Shift the signal +90 degrees
    newSignalFFT = signalFFT * cmath.rect( 1., shift )

    ## Reverse Fourier transform
    newSignal = np.fft.irfft(newSignalFFT)
    return newSignal




# %% test

#file = '/Users/goharshoukat/Documents/GitHub/Thesis_Tidal_Turbine/essais_ifremer_04_2021_backup/2021_04_hydrol_ECN/run159.txt'
#df, _ = access_file(file)
#angle, _ = angle_theta(df, 'Fx1')
#angle['signal with'] = scale(angle['signal'], with_mean=True, with_std= True, copy = True)
#angle['signal without'] = scale(angle['signal'], with_mean=True, with_std= False, copy = True)
#plt.plot(angle['theta'], angle['signal'])
#angle['result'] = angle['signal with'] - angle['signal without']
#polar_chart(df, 'Fx1')
