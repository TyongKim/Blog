"""
This scirpt is for polynomial linear regression.
'Auto.csv' dataset is used.

Created on Feb 28 2019

@author: taeyongkim
"""

# import libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Fetch dataset
# Since there are ? in the dataset, I ignore it
dataset = pd.read_csv('Auto.csv')
x = dataset.iloc[:,3].values # horsepower
y = dataset.iloc[:,0].values # mpg

x2 = []
y2 = []
for ii in range(len(x)):
    if x[ii] != '?':
        x2.append(float(x[ii]))
        y2.append(float(y[ii]))
    
del x, y
x2 = np.asarray(x2)
y2 = np.asarray(y2)

x2 = x2.reshape([len(x2),1])
y2 = y2.reshape([len(x2),1])

# Perform linear regression
lin_regressor = LinearRegression()
lin_regressor.fit(x2,y2)

# Perform polynomial
poly_reg = PolynomialFeatures(degree = 3)
x2_poly = poly_reg.fit_transform(x2)

poly_regressor = LinearRegression()
poly_regressor.fit(x2_poly, y2)

# Visualization - Linear regression
plt.figure()
plt.scatter(x2, y2, color='blue')
plt.plot(x2, lin_regressor.predict(x2), color='red')
plt.title('Prediction - Linear Regression')
plt.xlabel('Horsepower')
plt.ylabel('MPG')

# Visualization - Polynomial regression
x_plot = np.arange(np.min(x2), np.max(x2), 0.1)
x_plot = x_plot.reshape((len(x_plot), 1))
plt.figure()
plt.scatter(x2, y2, color='blue')
plt.plot(x_plot, poly_regressor.predict(poly_reg.fit_transform(x_plot)), color='red')
plt.title('Prediction - Polynomial Regression')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
