import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from model.utility import *
import copy

A = 2 * torch.eye(3)
b = 3 * torch.ones((3, 1))

def F(x):  # F(x) = G(x) + L(x)
    return torch.mm(x.t(), b) + torch.mm(x.t(), torch.mm(A, x))

def G(x):
    return torch.mm(x.t(), b) + 0.5 * torch.mm(x.t(), torch.mm(A, x))

def L(x):
    return 0.5 * torch.mm(x.t(), torch.mm(A, x))

xF = nn.Parameter(torch.ones((3, 1)), requires_grad = True)
optimizer = opt.Adam([xF])

print("Optimizing F ...")

n_iter = 6000   # find the minimum of F(x)
for i in range(n_iter):
    optimizer.zero_grad()
    func = F(xF)
    func.backward()
    optimizer.step()
    print("Iteration " + str(i) + " : ", xF, func.item())

print("Now, approximate the minimizer of G ...")

func = L(xF)
GL, HL = compute_derivative(func, [xF], hessian = True)  # cache the L'(xF) and L''(xF)

func = F(xF)
_, HF = compute_derivative(func, [xF], hessian = True)  # cache F''(xF)

xG = xF + torch.mm(torch.inverse(HF - HL), GL.view(-1, 1))  # Compute xG = argmin G(x) via Taylor approximation
print(xG.view(-1))  # xG = xF + (F''(xF) - L''(xF))^-1 L'(xF)
print(-torch.mm(torch.inverse(A), b).view(-1))  # closed-form solution for xG

