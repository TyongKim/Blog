"""
This script checks whether the Runge-Kutta 45 is properly working or not.
Developed by Taeyong Kim from the Seoul National University
Email: chs5566@snu.ac.kr
Mar 10, 2020
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, ode

#%% First example
def first_example(t, y):
    ydot = 2*y
    return ydot

# initial condition and time
y0 = [1]
t = np.linspace(0,3,300)

# Solve the problem
sol = solve_ivp(fun=first_example, t_span=(0,3), y0=y0, method="RK45" ,t_eval = t)
solution_t = sol.t
solution_y = sol.y

# analytic solution    
y_real = np.exp(2*t)
    
# Plot
plt.close('all')
plt.plot(t, y_real,'k-')
plt.plot(solution_t, solution_y.T, 'r--')
plt.legend(['Analytic solution', 'Numerical solution'])
#%% Second example

m = 1 # mass
c = 1 # damping
k = 1 # stiffness

ftt = np.arange(0,10,0.01)
ft = np.sin(ftt)  # External force

# initial parameters
y0 = [0, 0]

# Define the ordinary differential equation
def second_example(t,y, ftt, ft, m, c, k):
    
    y0, y1 = y
    
    ft2 = np.interp(t, ftt, ft)
    
    ydot_0 = y1
    ydot_1 = -c/m*y1 - k/m*y0 + ft2

    return [ydot_0, ydot_1]


sol = solve_ivp(lambda t, y:second_example(t,y, ftt, ft, m, c, k), t_span=(0,10), 
                y0=y0, method="RK45" ,t_eval = ftt)
solution_t = sol.t
solution_y = sol.y
    
# Plot
plt.figure()
plt.plot(solution_t, solution_y.T[:,0], 'r--')
plt.xlabel('t') 
plt.ylabel('u') 


