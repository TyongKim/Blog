"""
This scirpt is for K nearest neighborhood algrotihm which is one of the 
classification method.

'Heart.csv' dataset is used.

Created on Mar 15 2019

@author: taeyongkim
"""

# import libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fetch dataset
# The purpose is to predict heart disease based on age and the level of chol.
dataset = pd.read_csv('Heart.csv')
x1 = dataset.iloc[:,1].values # age
x2 = dataset.iloc[:,5].values # cholesterol
y = dataset.iloc[:,14].values # Yes or no

# Chage 'Yes' or 'No' to 1 and -1, respectively
yy = []
for ii in range(len(y)):
    if y[ii] == 'No':
        yy.append(0)
    else:
        yy.append(1)

y = np.asarray(yy)
X = np.c_[x1,x2]
del yy, x1, x2

# split the dataset, only 80% is used to training
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# perform classification: KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
# Prediction
y_pred = KNN.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Visulization
from matplotlib.colors import ListedColormap

X1, X2 = np.meshgrid(np.arange(np.min(X[:,0])-1,np.max(X[:,0])+1,0.05),
                     np.arange(np.min(X[:,1])-1,np.max(X[:,1])+1,0.2))

Z = KNN.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
plt.figure()
for ii in range(len(y)):
    if y[ii] == 0:
        plt.scatter(X[ii , 0], X[ii, 1], c='r')
    else:
        plt.scatter(X[ii , 0], X[ii, 1], c='b')

plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend(['No','Yes'])

plt.figure()
plt.contourf(X1,X2,Z,cmap = ListedColormap(('red', 'blue')))
plt.xlabel('Age')
plt.ylabel('Cholesterol')