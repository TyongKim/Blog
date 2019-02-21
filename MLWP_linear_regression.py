"""
This scirpt is for simple linear regression.
'Auto.csv' dataset is used.

Created on Feb 21 2019

@author: taeyongkim
"""

# import libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Fetch dataset
# Since there are ? in the dataset, I ignore it
dataset = pd.read_csv('Auto.csv')
x = dataset.iloc[:,3].values
y = dataset.iloc[:,0].values

x2 = []
y2 = []
for ii in range(len(x)):
    if x[ii] != '?':
        x2.append(float(x[ii]))
        y2.append(float(y[ii]))
    
del x, y
x2 = np.asarray(x2)
y2 = np.asarray(y2)
# Splitting the dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(x2,y2, 
                                                    test_size = 1/5, 
                                                    random_state = 117)

# Perform linear regression
regressor = LinearRegression()
regressor.fit(X_train.reshape([len(X_train),1]), Y_train.reshape([len(X_train),1]))

# Prediction using test dataset
Y_pred = regressor.predict(X_test.reshape([len(X_test),1]))

plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train.reshape([len(X_train),1])), color='red')
plt.xlabel('Horsepower')
plt.ylabel('MPG')