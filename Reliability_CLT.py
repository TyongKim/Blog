#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:17:52 2021

@author: taeyongkim
"""

# Central Limit Theorem
import numpy as np
import matplotlib.pyplot as plt

total =1000; mu=3; sigma=1;
avg_2 = []; avg_10 = []; avg_50 = []; avg_100 = []; avg_1000 = [];
for ii in range(total):
    tmp_2 = np.random.lognormal(mu, sigma,2)
    tmp_10 = np.random.lognormal(mu, sigma,10)
    tmp_50 = np.random.lognormal(mu, sigma,50)
    tmp_100 = np.random.lognormal(mu, sigma,100)
    tmp_1000 = np.random.lognormal(mu, sigma,1000)
    
    avg_2.append(np.average(tmp_2))
    avg_10.append(np.average(tmp_10))
    avg_50.append(np.average(tmp_50))
    avg_100.append(np.average(tmp_100))
    avg_1000.append(np.average(tmp_1000))

plt.close('all'); plt.figure()
plt.subplot(151); plt.hist(avg_2)
plt.subplot(152); plt.hist(avg_10)
plt.subplot(153); plt.hist(avg_50)
plt.subplot(154); plt.hist(avg_100)
plt.subplot(155); plt.hist(avg_1000)