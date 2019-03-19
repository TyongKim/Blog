"""
This scirpt is for K nearest neighborhood algrotihm which is one of the 
classification method.

A hypothetic dataset is used.

Created on Mar 19 2019

@author: taeyongkim
"""

# import libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make dataset
x1 = np.random.normal(-2, 1.5, 500)
y1 = np.zeros([500,])
x2 = np.random.normal(2, 1.5, 500)
y2 = np.ones([500,])

X = np.r_[x1,x2]
Y = np.r_[y1,y2]

X = X.reshape(len(X),1)
#Y = X.reshape(len(Y),1)
del x1, y1, x2, y2

# Divide train and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# train dataset using logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 5)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Plot the sigmoid function

plt.close('all')
plt.scatter(X_test,y_test)
plt.scatter(X_test,y_pred)
