"""
This scirpt is for generating artificial ground motion based on Bi and 
Hao (2012). Modelling and simulation of spatially varying earthquake ground 
motions at sites with varying conditions.
Using sepctral representation and Kanai Tajimi model

Created on Feb 20 2019

@author: taeyongkim
"""
#%% Section 4.2 Example 1 PSD compatible ground motion simulation
# import libararies
import numpy as np
import matplotlib.pyplot as plt

#%% Input definition - should be changed
# central frequency of KT
wg = 5*2*np.pi
# damping ratio of KT
zg = 0.6
# cnetral frequency of high pass filter
wf = 0.25*2*np.pi
# damping ratio of high pass filter
zf = 0.6
# The parmeters of horizontal motion
Gamma_KT = 0.0034 # m2/s3
dw = 25*2*np.pi/2048 # Frequency step; cut-off freq = 25Hz, N=2048
w = np.arange(dw,25*2*np.pi+dw,dw)  # frequnecy
n = len(w)
dt = 0.01
t = np.arange(0,30,dt) # Total duration

#%%
# Base rock motion - Filtered Kanai-Tajimi power spectrum density
# Kanai tagimi
PSD_KT = Gamma_KT*(wg**4+4*zg**2*wg**2*w**2)/((wg**2-w**2)**2+4*zg**2*wg**2*w**2)
# High pass filter
PSD_attenu = (w**4)/((wf**2-w**2)**2+4*zf**2*wf**2*w**2) 
PSD=PSD_KT*PSD_attenu

# Jenning's envelop function, t_0 = 2sec and t_n = 10sec
A = []
for ii in range(len(t)):
    if t[ii]<2:
        A.append((t[ii]/2)**2)
    elif t[ii]<10:
        A.append(1)
    else:
        A.append(np.exp(-0.155*(t[ii]-10)))
A = np.asarray(A)

# Plot Kani Tajimi Power spectral density given frequency
plt.close('all')
plt.figure()
plt.plot(w/2/np.pi,PSD)
plt.xlabel('Frequency')
plt.xlim([0,30])

#%% Spectral representation
theta = np.random.uniform(0,2*np.pi,len(w)) # % Random phases (uniform)

# w_n represents an upper cut-off frequency = 25 Hz (=157.0796 rad/s)
# already considered
Xs = np.zeros([len(t),1])
for ii in range(len(w)):
    Xs = Xs + (2*np.sqrt(PSD[ii]*dw)*
               np.cos(w[ii,]*t+theta[ii])).reshape(len(t),1)

Xs = A.reshape(len(t),1)*Xs
plt.figure()
plt.plot(t,Xs)

