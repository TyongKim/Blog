"""
This scirpt performs the polynomial regression by using the PyTorch

Created on March 1 2020

Developed by Taeyong Kim from the Seoul National University

"""

# import libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fetch dataset
# Since there are ‘?’ in the dataset, I ignore it
# Final input and output dataset that we use in the regression analysis are x2 and y2, respectively
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

# Define functions 
class PolyRegression(nn.Module):  # inheriting from nn.Module

    def __init__(self, n_features):
        super(PolyRegression, self).__init__()

        self.linear = nn.Linear(in_features=n_features, out_features=1,
                                bias=True)
        
    def forward(self, x):
        return self.linear(x)


def MakePoly(x, degree):
    x = torch.FloatTensor(x)
    # Generate polynomials such as [1, x, x^2, x^3, x^4].
    polynomials = torch.cat([x ** i for i in range(1, degree+1)], 1)
    
    return polynomials

# Preprocess the dataset, standardization is adopted
degree = 3
New_x2 = MakePoly(x2, degree)  # Generate polynomials for the training
New_y2 = torch.FloatTensor(y2) # Change to Numpy to Torch tensor
check_New_x2 = New_x2.numpy()  # check in the variable explorer by changing Pytorch to Numpy

means_x2 = New_x2.mean(dim=0, keepdim=True)
stds_x2 = New_x2.std(dim=0, keepdim=True)
standardized_New_x2 = (New_x2 - means_x2) / stds_x2
means_y2 = New_y2.mean(dim=0, keepdim=True)
stds_y2 = New_y2.std(dim=0, keepdim=True)
standardized_New_y2 = (New_y2 - means_y2) / stds_y2


# Define model and optimizer, Adam optimizer is employed
model = PolyRegression(degree)        
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Care needs to be taken when determining learning rate

# Training parameters
nb_epochs = 1000
for epoch in range(1, nb_epochs+1):
    
    # linear regression function
    prediction = model(standardized_New_x2)

    loss = F.mse_loss(prediction, standardized_New_y2) # Loss function (MSE)
    
    optimizer.zero_grad() # initialize by 0 gradient
    loss.backward()       # perform backward processing to estimate gradient
    optimizer.step()      # carry out gradient descent
    
    # Print the resutls every 100 steps
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(
            epoch, nb_epochs, loss.item() ))    
       
        
# Dataset for Visualization 
x_plot = np.arange(np.min(x2), np.max(x2), 0.1)
x_plot = x_plot.reshape((len(x_plot), 1))

# Predict and Recover the dataset for visualization
New_x_plot = MakePoly(x_plot, degree)
standardized_x_plot = (New_x_plot - means_x2) / stds_x2
standardized_y_plot = model(standardized_x_plot)
y_plot = standardized_y_plot*stds_y2 +means_y2

# Plotting
plt.close('all')
plt.figure()
plt.scatter(x2, y2, color='blue')
plt.plot(x_plot, y_plot.detach().numpy(), color='red')
plt.title('Prediction - Polynomial Regression')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
    