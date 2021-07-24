#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:29:17 2021

@author: goharshoukat
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import re

def load_ldv3D(name, ang):
    if ang==0:
        M = np.array([[0.005247, 1.004515, -0.005563], [0.999819, -0.013590, 0.008014], [-0.038989, -1.412896, 1.734007]])
        y0 = 1800
        z0 = 700
        
    elif ang == 90:
         M=np.array([[0.005247, 1.004515, -0.005563], [0.038989, 1.412896, -1.734007], [0.999819, -0.013590, 0.008014]])
         y0 = 1210
         z0 = 1290
    else:
        print('The angle must be 0 or 90 degrees only')
    
    file = open(name)
    #extract information about the xyz positioning of the probes
    line = file.readlines()[3]
    file.close()
    #removes the colon and mm from the str and splits it into components
    xyz = re.split(';|mm', line)
    #removes the white spaces
    xyz = [x.strip('  ') for x in xyz]
    #gets rid of the additional elements created due to split at mm and ;
    xyz= [x for x in xyz if x][:4] #an additional element was created due to \n encoutered during split. [:4] eleminates that additional character
    xyz = list(map(float, xyz)) #converts strings to floating points
    
    #convert mm to m
    x = xyz[1]/1e3
    y = (xyz[2] - y0)/1e3
    z = (xyz[3] - z0)/1e3
   
    #read data file
    data = pd.read_csv(name, sep = '\t', skiprows=5)
    data = data.drop(columns = ['Row#'])
    
    a1 = data['Transit Time [us]']
    a2 = data['Transit Time{G2} [us]']
    a3 = data['Transit Time{G3} [us]']
    
    i1 = (a1!=0)
    i2 = (a2!=0)
    i3 = (a3!=0)
    
    u1 = data['Velocity U [m/s]'][i1]
    u2 = data['Velocity V{G2} [m/s]'][i2]
    u3 = data['Velocity W{G3} [m/s]'][i3]

    #convert to seconds
    t1 = (data['Arrival Time [ms]'][i1]/1000).rename('Arrival Time [s]')
    t2 = (data['Arrival Time{G2} [ms]'][i2]/1000).rename('Arrival Time{G2} [s]')
    t3 = (data['Arrival Time{G3} [ms]'][i3]/1000).rename('Arrival Time{G3} [s]')
    
    tmin = np.min(np.array([[min(t1), min(t2), min(t3)]]))
    tmax = np.max(np.array([[max(t1), max(t2), max(t3)]]))
    
    n = np.min(np.array([[len(t1), len(t2), len(t3)]]))
    t= np.linspace(tmin, tmax, n)
    
    u1i = interp1d(t1, u1, fill_value = 'extrapolate')(t)
    u2i = interp1d(t2, u2, fill_value = 'extrapolate')(t)
    u3i = interp1d(t3, u3, fill_value = 'extrapolate')(t) 
    
    lda = np.array([u1i, u2i, u3i])
    U = np.dot(M, lda).T
    
    u = U[:, 0]
    v = U[:, 1]
    w = U[:, 2]

    #function returns time and u array only as required
    return t, u
#%% test fucnction section
'''
import matplotlib.pyplot as plt
file = open('run034.000001.txt')


line = file.readlines()[3]
    #removes the colon and mm from the str and splits it into components
file.close()

xyz = re.split(';|mm', line)
    #removes the white spaces
xyz = [x.strip('  ') for x in xyz]
    #gets rid of the additional elements created due to split at mm and ;
xyz= [x for x in xyz if x][:4] #an additional element was created due to \n encoutered during split. [:4] eleminates that additional character
xyz = list(map(float, xyz))

data = pd.read_csv('run034.000001.txt', sep = '\t', skiprows = 5)
#data = data.drop(columns = ['Row#'])    


x = xyz[1]/1e3
y = (xyz[2])/1e3
z = xyz[3]/1e3

a1 = data['Transit Time [us]']
a2 = data['Transit Time{G2} [us]']
a3 = data['Transit Time{G3} [us]']

i1 = (a1!=0)
i2 = (a2!=0)
i3 = (a3!=0)

u1 = data['Velocity U [m/s]'][i1]
u2 = data['Velocity V{G2} [m/s]'][i2]
u3 = data['Velocity W{G3} [m/s]'][i3]

#convert to seconds
t1 = (data['Arrival Time [ms]'][i1]/1000).rename('Arrival Time [s]')
t2 = (data['Arrival Time{G2} [ms]'][i2]/1000).rename('Arrival Time{G2} [s]')
t3 = (data['Arrival Time{G3} [ms]'][i3]/1000).rename('Arrival Time{G3} [s]')

tmin = np.min(np.array([[min(t1), min(t2), min(t3)]]))
tmax = np.max(np.array([[max(t1), max(t2), max(t3)]]))

n = np.min(np.array([[len(t1), len(t2), len(t3)]]))
t= np.linspace(tmin, tmax, n)

u1i = interp1d(t1, u1, fill_value= 'extrapolate')(t)
u2i = interp1d(t2, u2, fill_value = 'extrapolate')(t)
u3i = interp1d(t3, u3, fill_value = 'extrapolate')(t) 
M=np.array([[0.005247, 1.004515, -0.005563], [0.038989, 1.412896, -1.734007], [0.999819, -0.013590, 0.008014]])
y0 = 1210
z0 = 1290

lda = np.array([u1i, u2i, u3i])
U = np.dot(M, lda).T

u = U[:, 0]
v = U[:, 1]
w = U[:, 2]

plt.plot(t1, u1)


name = '/Users/goharshoukat/Documents/GitHub/Thesis_Tidal_Turbine/essais_ifremer_04_2021/LDV/2021_04_hydrol_ECN/renamed/LDV0421V08TSR25P594SR1.txt'
t2, u2 = load_ldv3D(name, 0)

plt.plot(t2, u2)

'''
