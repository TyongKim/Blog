"""
This scirpt illustrates the basic commands of the pytorch.

Created on Feb 17 2020

Developed by Taeyong Kim from the Seoul National University

"""


# 0. import libraries
import numpy as np
import torch 

# 1. Find the dimension of tensors
ty = torch.FloatTensor([[1,2,3,4,5]])  # tensor whose dimension is 1x5
print(ty.dim())  # find the dimension of the tensor
print(ty.size()) # find the size of the tensor
print(ty[:,2:-1])  # slicing the tensor


# 2. Summation and multiplication
JS = torch.FloatTensor([[1,2]])  # tensor whose dimension is 1x2
TK = torch.FloatTensor([[4,5]])  # tensor whose dimension is 1x2
TK2 = torch.FloatTensor([4])     # tensor whose dimension is 1x0
JS2 = torch.FloatTensor([[2],[3]])

# 2.1 Summation
print(JS+TK)
print(JS+TK2)  # Broadcasting semantic example 1
print(JS+JS2)  # Broadcasting semantic example 2

# 2.2 Multiplication
print(JS*TK)   # Element wise multiplication
print(JS*JS2)  # Boradcasting semantic example 3
print(JS.matmul(JS2))  # Matrix multiplication


# 3. Reshape tensors
ty = torch.FloatTensor([[[1,2,3], [2,3,4]],
                        [[10,11,12], [11,12,13]]])

# 3.1 reshape tensor using view commands
print(ty.size())       # The size of ty is (2,2,3)
print(ty.view(-1,3))   # Make ty to (??,3), where ??=2*2

# 3.2 squeeze and Unsqeeze
JS0 = torch.FloatTensor([1,2])  # tensor whose dimension is 2
print(JS0.size())
print(JS0.unsqueeze(1))
print(JS0.unsqueeze(1).size())
print(JS0.unsqueeze(0))
print(JS0.unsqueeze(0).size())


# 4. Stacking
JS = torch.FloatTensor([1,2])  # tensor whose dimension is 2
TK = torch.FloatTensor([4,5])  # tensor whose dimension is 2
JS2 = torch.FloatTensor([[2],[3]]) # tensor whose dimension is 2x1

print(torch.stack([JS, TK]))
print(torch.stack([JS, TK], dim=1))
print(torch.cat([JS.unsqueeze(1), TK.unsqueeze(1), JS2], dim=0))


# 5. Ones and zeros
JS_TK = torch.FloatTensor([[1, 2,3 ], [4, 5, 6]])
print(JS_TK.size())
print(torch.ones_like(JS_TK))
print(torch.zeros_like(JS_TK))
