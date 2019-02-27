"""
This scirpt is for multiple linear regression.
'Auto.csv' dataset is used.

Created on Feb 27 2019

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
x1 = dataset.iloc[:,3].values    # horsepower
x2 = dataset.iloc[:,4].values    # weight
x3 = dataset.iloc[:,7].values    # origin
y = dataset.iloc[:,0].values     # mpg


# Omit the missing data
x1_2 = []
x2_2 = []
x3_2 = []
y_2 = []
for ii in range(len(x1)):
    if x1[ii] != '?':
        x1_2.append(float(x1[ii]))
        x2_2.append(float(x2[ii]))
        x3_2.append(x3[ii])
        y_2.append(float(y[ii]))
    
del x1, x2, x3, y

x1_2 = np.asarray(x1_2)
x2_2 = np.asarray(x2_2)
x3_2 = np.asarray(x3_2)
y_2 = np.asarray(y_2)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x3_2 = labelencoder.fit_transform(x3_2)
onehotencoder = OneHotEncoder(categorical_features = [0])
x3_2 = onehotencoder.fit_transform(x3_2.reshape([len(x3_2),1])).toarray()

# Final dataset
X = np.c_[x1_2,x2_2,x3_2]

# Splitting the dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,y_2, 
                                                    test_size = 1/5, 
                                                    random_state = 117)

# Perform linear regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train.reshape([len(X_train),1]))

# Prediction using test dataset
Y_pred = regressor.predict(X_test)

