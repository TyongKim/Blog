# This script is to calculate spectral responses of an SDOF system
# Developed by Taeyong kim from the Seoul National University
# April 5, 2020
# chs5566@snu.ac.kr

###############################################################################
# Input discription
# Newmark's method: linear systems by Chopra 3rd edition pp.177
# m: mass
# k: stiffness
# xi: damping ratio
# GM: ground motion history
# dt_analy: time step
# gamma & beta are the parameter of Newmark's method

# output discription
# dis: displacement response 
# vel: velocity response
# acc: acceleration response
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

def Newmark_TK(m, k, xi, GM, dt,dt_analy, gamma, beta):
    
    # Import a library
    
   
    temp_GM = GM
    num_data = len(temp_GM)
    tmp = np.arange(0,num_data)*dt
    tmp2 = np.arange(0,num_data*dt/dt_analy)*dt_analy
    GM = np.interp(tmp2,tmp,temp_GM)
    
    # Initial calculations
    c = 2*xi*np.sqrt(m*k)
    k_hat = k+gamma/beta/dt_analy*c+1/beta/dt_analy**2*m
    a = 1/beta/dt_analy*m+gamma/beta*c
    b = 1/2/beta*m+dt_analy*(gamma/2/beta-1)*c
    p = -m*GM
    
    # Assume initial acceleration,     
    results = np.zeros([len(GM),3])

    
    # We assume initial velocitiy and displacement are zero
    dis = 0
    vel = 0
    pi = 0      # Assume initial force is zero
    acc = (pi-c*vel-k*dis)/m

    
    # Calculations for each time step, i
    for i in range(len(GM)):
        dp = p[i]-pi
        dp_hat = dp + a*vel + b*acc
        
        ddis = dp_hat/k_hat
        dvel = gamma/beta/dt_analy*ddis - gamma/beta*vel +dt_analy*(1-gamma/2/beta)*acc
        dacc = 1/beta/(dt_analy*dt_analy)*ddis - 1/beta/dt_analy*vel -1/2/beta*acc
        
        dis = dis+ddis
        vel = vel+dvel
        acc = acc+dacc
        
        results[i,0] = dis
        results[i,1] = vel
        results[i,2] = acc + GM[i]
        pi = p[i]
        
    return results


# Example
g = 9.8#; % ground acceleration m/s2

dt = 0.005
xgt = np.loadtxt('GM0.txt') # Import ground motion dataset (unit: g)
xgt = xgt*g

dw = 0.1
w = np.arange(dw,20*2*np.pi+dw,dw)  # frequnecy

Spectral_acce = []
Response_period = []
for jj in range(len(w)):
    period = 2*np.pi/w[jj]
    if period<=6:           
        results = Newmark_TK(1, w[jj]**2, 0.05, xgt, dt, dt, 1/2, 1/6)
        Spectral_acce.append(np.max(np.abs(results[:,2]))) # Spectral acceleration
        Response_period.append(period)

plt.figure()        
plt.plot(Response_period, np.asarray(Spectral_acce)/g)
plt.xlabel('Period, sec')
plt.ylabel('spectral acceleration, g')