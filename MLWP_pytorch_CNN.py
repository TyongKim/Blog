"""
This scirpt shows an example of convolutional neural network using PyTorch

Created on April 10 2020

Developed by Taeyong Kim from the Seoul National University

"""


# import libraries
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt


# If GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters of training epoch and batch size
training_epochs = 20
batch_size = 128

# Load MNIST dataset, original image has 0-255 ==> pytorch has 0-1
# If there is no dataset in root file, then download the datasets
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform= transforms.ToTensor(), 
                          download=True)
mnist_test = dsets.MNIST(root="MNIST_data/", train=False, transform= transforms.ToTensor(),
                         download=True)

# Load datset to Python, total dataset is 60,000
train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size,
                                     shuffle=True, drop_last=False)


# MNIST data image of shape 28 * 28 = 784
# Make a general DNN model
class no_CNN_model(nn.Module):
    def __init__(self):
        super(no_CNN_model, self).__init__()
        self.layer1 = nn.Linear(28*28,512, bias=True)
        self.layer2 = nn.Linear(512,256, bias=True)
        self.layer3 = nn.Linear(256,10, bias=True)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.ReLU(output)
        output = self.layer2(output)
        output = self.ReLU(output)
        output = self.layer3(output)
        return output

# Make a CNN model
"""
When you want to know the size of each layer, I highly recommend to use the below
procedures.

input_abc = torch.randn(1,1,28,28)
abc = nn.Conv2d(1,16,kernel_size=3,  padding=1)
abc2 = nn.MaxPool2d(2)
bcd = nn.Conv2d(16,32,kernel_size=3,  padding=1)
bcd2 = nn.MaxPool2d(2)
output = abc(input_abc)
print(output.shape)
output = abc2(output)
print(output.shape)
output = bcd(output)
print(output.shape)
output = bcd2(output)
print(output.shape)
"""

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2)
                                        )
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2)
                                        )
        
        self.FC_layer1 = nn.Linear(32*7*7,10, bias=True)
        
    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = output.view(output.size(0),-1) # Flattening
        output = self.FC_layer1(output)
        return output
    
model_noCNN = no_CNN_model().to(device)    
model_CNN = CNN_model().to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer_noCNN = torch.optim.Adam(model_noCNN.parameters())
optimizer_CNN = torch.optim.Adam(model_CNN.parameters())

# Start training
loss_train = []
loss_test = []
for epoch in range(training_epochs):
    avg_cost_noCNN = 0
    avg_cost_CNN = 0
    total_batch = len(train_loader)
    for X, Y in train_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X_no_CNN = X.view(-1, 28 * 28).to(device)
        X_CNN = X.to(device)
        Y = Y.to(device)
        
        # train model no CNN
        optimizer_noCNN.zero_grad()
        cost1 = criterion(model_noCNN(X_no_CNN), Y)
        cost1.backward()
        optimizer_noCNN.step()
        
        avg_cost_noCNN += cost1 / total_batch
        
        # train model CNN
        optimizer_CNN.zero_grad()
        cost2 = criterion(model_CNN(X_CNN), Y)
        cost2.backward()
        optimizer_CNN.step()
        
        avg_cost_CNN += cost2 / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 
          'train_cost_noCNN =', '{:.9f}'.format(avg_cost_noCNN),
          'train_cost_CNN =', '{:.9f}'.format(avg_cost_CNN))

print('Training finished')

# Test the dataset
with torch.no_grad():
    X_test_noCNN = mnist_test.data.view(-1,28*28).float().to(device)
    X_test_CNN = mnist_test.data.view(len(mnist_test),1,28,28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction1 = model_noCNN(X_test_noCNN)
    prediction2 = model_CNN(X_test_CNN)
    
    correct_prediction1 = torch.argmax(prediction1, 1) == Y_test
    correct_prediction2 = torch.argmax(prediction2, 1) == Y_test

    accuracy1 = correct_prediction1.float().mean()
    accuracy2 = correct_prediction2.float().mean()
    
    print('Accuracy_noCNN:', accuracy1.item(),
          'Accuracy_CNN:', accuracy2.item())
    
print('Test evaluation finished.')    
