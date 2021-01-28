#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:24:43 2021

@author: taeyongkim
"""


import numpy as np
import matplotlib.pyplot as plt

# define domain
dx = 0.001
L = np.pi
x = L * np.arange(-1+dx,1+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))

# Define hat function
f = np.zeros_like(x)
f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)

plt.close('all')
fig, ax = plt.subplots()
ax.plot(x,f,'-',color='k',LineWidth=2)

period = 2*L; angular_freq = 2*np.pi/period

# Compute Fourier series
A0 = np.sum(f * np.ones_like(x))/len(f)
fFS = A0
A = []

for k in range(20):
    tmp_A = np.sum(f * np.exp(-1j*(k+1)*angular_freq*x)) / len(f)
    # Inner product
    fFS = fFS + np.real(tmp_A*np.exp(1j*(k+1)*angular_freq*x) + np.conj(tmp_A)*np.exp(-1j*(k+1)*angular_freq*x))
    ax.plot(x,fFS,'-')
    A.append(tmp_A)
