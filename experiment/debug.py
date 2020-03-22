import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from model.utility import *

A = torch.eye(10)
A[0, 0] += 100
b = torch.ones((10, 1))

x = nn.Parameter(torch.ones((10, 1)), requires_grad = True)
z = nn.Parameter(2 * torch.ones((10, 1)), requires_grad = True)
var = torch.cat([x, z])
F = torch.mm(x.t(), b) + torch.mm(x.t(), torch.mm(A, x)) + torch.mm(z.t(), b)

G, H = compute_derivative(F, var, hessian = True)

