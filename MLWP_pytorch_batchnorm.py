"""
This scirpt shows an example of a batch normalization using PyTorch

Created on April 7 2020

Developed by Taeyong Kim from the Seoul National University

"""


# import libraries
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch

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
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size,
                                     shuffle=True, drop_last=False)
# MNIST data image of shape 28 * 28 = 784
# Model w/ Batch normalization
linear1 = torch.nn.Linear(784, 256, bias=True).to(device)
linear2 = torch.nn.Linear(256, 64, bias=True).to(device)
linear3 = torch.nn.Linear(64, 10, bias=True).to(device)

relu = torch.nn.ReLU().to(device)
bn1 = torch.nn.BatchNorm1d(256).to(device)
bn2 = torch.nn.BatchNorm1d(64).to(device)

model1 = torch.nn.Sequential(linear1, bn1, relu,
                             linear2, bn2, relu,
                             linear3).to(device) # Define the DNN model (w/ BN)

# Model w/o Batch normalization
linear21 = torch.nn.Linear(784, 256, bias=True).to(device)
linear22 = torch.nn.Linear(256, 64, bias=True).to(device)
linear23 = torch.nn.Linear(64, 10, bias=True).to(device)

model2 = torch.nn.Sequential(linear1, relu,
                             linear2, relu,
                             linear3).to(device) # Define the DNN model (w/o BN)

# define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

# Start training
loss_train = []
loss_test = []
for epoch in range(training_epochs):

    model1.train()  # For training the model
    total_batch = len(train_loader)
    for X, Y in train_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)
        
        # train model w/ BN
        optimizer1.zero_grad()
        cost1 = criterion(model1(X), Y)
        cost1.backward()
        optimizer1.step()

        # train model w/o BN
        optimizer2.zero_grad()
        cost2 = criterion(model2(X), Y)
        cost2.backward()
        optimizer2.step()
        
     
    with torch.no_grad():
        model1.eval()     # For predicting the model
        
        avg_cost1_tr = 0
        avg_cost2_tr = 0           
        for X, Y in train_loader:
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            cost1 = criterion(model1(X), Y)
            avg_cost1_tr += cost1 / total_batch
            
            cost2 = criterion(model2(X), Y)
            avg_cost2_tr += cost2 / total_batch

        avg_cost1_te = 0
        avg_cost2_te = 0           
        for X, Y in test_loader:
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            cost1 = criterion(model1(X), Y)
            avg_cost1_te += cost1 / total_batch
            
            cost2 = criterion(model2(X), Y)
            avg_cost2_te += cost2 / total_batch            
            
        loss_train.append([avg_cost1_tr, avg_cost2_tr])
        loss_test.append([avg_cost1_te, avg_cost2_te])

    print('Epoch:', '%04d' % (epoch + 1), 
          'train_cost_w/BN =', '{:.9f}'.format(avg_cost1_tr),
          'train_cost_w/oBN =', '{:.9f}'.format(avg_cost2_tr),
          'test_cost_w/BN =', '{:.9f}'.format(avg_cost1_te),
          'test_cost_w/oBN =', '{:.9f}'.format(avg_cost2_te))
    

print('Learning finished')
