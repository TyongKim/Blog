"""
This scirpt performs the XOr problem by using the PyTorch

Created on March 5 2020

Developed by Taeyong Kim from the Seoul National University

"""

# import libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# Make the dataset for the XOr problem
x_input = torch.FloatTensor([[0,0], [1,0], [0,1], [1,1]])
y_output = torch.FloatTensor([[0],[1],[1],[0]])

# Make a hyper plane
hyperplane = torch.nn.Linear(2,1)
activ_sigm = torch.nn.Sigmoid()
model = torch.nn.Sequential(hyperplane, activ_sigm)

# Loss and optimizer are defined
criterion = torch.nn.BCELoss() # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters()) # adam optimizer

# Start training the parameters
nb_epochs = 5000
for epoch in range(1, nb_epochs+1):
    
    # linear regression function
    prediction = model(x_input)

    loss = criterion(prediction, y_output) # Loss function
    
    optimizer.zero_grad() # initialize by 0 gradient
    loss.backward()       # perform backward processing to estimate gradient
    optimizer.step()      # carry out gradient descent
    
    # Print the resutls every 100 steps
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(
            epoch, nb_epochs, loss.item() ))    

# Calculate the accruacy of the trained model
with torch.no_grad():
    tmp_y_estimate = model(x_input) # prediction from the model
    # if the pedicted value is greater than 0.5, it is True. Otherwise, False.
    y_estimate = (tmp_y_estimate>0.5).float() 
    # accuarcy is calculated
    accuracy = (y_estimate == y_output).float().mean()
    # display the output
    print('Predicted values: ', tmp_y_estimate.detach().numpy(),
          '\nAccuracy: ', accuracy.detach().numpy())
    