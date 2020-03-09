"""
This scirpt performs the backpropagation by using the PyTorch

Created on March 6 2020

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

# Make a hyper parameters
W1 = torch.nn.Parameter(torch.Tensor(2, 2))
b1 = torch.nn.Parameter(torch.Tensor(2))
W2 = torch.nn.Parameter(torch.Tensor(2, 1))
b2 = torch.nn.Parameter(torch.Tensor(1))

torch.nn.init.normal_(W1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(W2)
torch.nn.init.normal_(b2)

lr = 0.01 # learning rate

# Define sigmoid functions
def sigmoid(ty):
    sig = 1.0/(1.0+torch.exp(-ty))
    return sig

# Define derivative of sigmoid
def sigmoid_derivative(ty):
    sig_der = sigmoid(ty)*(1.0-sigmoid(ty))
    return sig_der

criterion = torch.nn.BCELoss() # Binary Cross Entropy

# Start training the parameters
nb_epochs = 20000
for epoch in range(1, nb_epochs+1):
    
    # Forward propagation
    f1 = torch.add(torch.matmul(x_input, W1), b1) # First layer
    a1 = sigmoid(f1) # activation function, 4*2
    f2 = torch.add(torch.matmul(a1, W2), b2) # Second layer
    a2 = sigmoid(f2) # activation function, 4*1
    
    # Estimate loss, Binary Cross Entropy is used
    loss1 = -torch.mean(y_output*torch.log(a2)+(1-y_output)*torch.log(1-a2))

    # Back propagation
    # derivative of the loss function, small value is introduced to prevent the divergence 
    d_L_a2 = (a2-y_output)/(a2*(1-a2)+0.0000001) # 4*1
    
    d_L2_b2 = sigmoid_derivative(f2)*1*d_L_a2 # 4*1
    d_L2_w2 = torch.matmul(torch.transpose(a1,0,1), sigmoid_derivative(f2)*d_L_a2) # 2*1
    d_L2_a1 = torch.matmul(sigmoid_derivative(f2)*d_L_a2, torch.transpose(W2,0,1)) # 4*2
    
    d_L2_b1 = sigmoid_derivative(f1)*1*d_L2_a1 # 4*2
    d_L2_w1 = torch.matmul(torch.transpose(x_input,0,1),sigmoid_derivative(f1)*d_L2_a1) # 2*2
    
    # Update the weigths
    W2 = W2 - lr*d_L2_w2
    b2 = b2 - lr*torch.mean(d_L2_b2,0)
    W1 = W1 - lr*d_L2_w1
    b1 = b1 - lr*torch.mean(d_L2_b1,0) 


    # Print the resutls every 100 steps
    if epoch % 1000 == 0:
        print('Home-made function, Epoch {:4d}/{} Loss1: {:.6f}'.format(
            epoch, nb_epochs, loss1.item() ))    
        
        
        
#%% Using the predefined functions
    
# Make the dataset for the XOr problem
x_input = torch.FloatTensor([[0,0], [1,0], [0,1], [1,1]])
y_output = torch.FloatTensor([[0],[1],[1],[0]])

# Make a hyper plane
hyperplane = torch.nn.Linear(2,2)
hyperplane2 = torch.nn.Linear(2,1)
activ_sigm = torch.nn.Sigmoid()
model = torch.nn.Sequential(hyperplane,activ_sigm, hyperplane2, activ_sigm)

# Loss and optimizer are defined
criterion = torch.nn.BCELoss() # Binary Cross Entropy
#optimizer = torch.optim.Adam(model.parameters()) # adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Start training the parameters
nb_epochs = 20000
for epoch in range(1, nb_epochs+1):
    
    # linear regression function
    prediction = model(x_input)

    loss = criterion(prediction, y_output) # Loss function
    
    optimizer.zero_grad() # initialize by 0 gradient
    loss.backward()       # perform backward processing to estimate gradient
    optimizer.step()      # carry out gradient descent
    
    # Print the resutls every 100 steps
    if epoch % 1000 == 0:
        print('Built-in function, Epoch {:4d}/{} Loss: {:.6f}'.format(
            epoch, nb_epochs, loss.item() ))          
        
        
        
        
        