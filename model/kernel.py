import torch
import torch.nn as nn
from model.utility import *
import numpy as np

class CovFunction(nn.Module):
    def __init__(self, n_dim, scales = None, c = torch.tensor(0.0), opt = True):
        super(CovFunction, self).__init__()
        self.n_dim = n_dim

        if scales is None:
            #self.weights = torch.tensor(np.random.randn(self.n_dim, 1))
            self.weights = nn.Parameter(torch.tensor(np.random.randn(self.n_dim, 1)),
                                        requires_grad = opt)
        else:
            #self.weights = scales
            self.weights = nn.Parameter(scales, requires_grad = opt)
        if type(c) != torch.Tensor:
            c = torch.tensor(c)
        self.sn = nn.Parameter(c, requires_grad = opt)

    def forward(self, U, V = None):  # U is of size m by n_dim, V is of size n by n_dim
        if V is None:
            V = U
        assert (len(U.size()) == 2) and (len(V.size()) == 2), "Input matrices must be 2D"
        assert U.size(1) == V.size(1), "Input matrices must agree on the second dimension"
        diff = V[None, :, None, :] - U[:, None, :, None]  # m by n by d by d
        scales = torch.exp(-1.0 * self.weights)
        scales = scales * torch.eye(self.n_dim).to(device)  # diagonal by by d -- length-scale on diag
        cov = (diff * scales[None, None, :, :]) ** 2   # m by n by d by d @ 1 by 1 by d by d
        cov = torch.exp(-0.5 * torch.sum(cov, dim = [2, 3]))  # m by n by d by d ==> m by n after sum
        return torch.exp(2.0 * self.sn).to(device) * cov


class MeanFunction(nn.Module):  # simply a constant mean function
    def __init__(self, c = torch.tensor(1.0), opt = True):
        super(MeanFunction, self).__init__()
        if type(c) != torch.Tensor:
            c = torch.tensor(c)
        self.mean = nn.Parameter(c, requires_grad = opt)

    def forward(self, U):  # simply the constant mean function; input form: (_, x_dim)
        assert len(U.size()) == 2, "Input matrix must be 2D"
        n = U.size(0)
        return torch.ones(n, 1).to(device) * self.mean


class LikFunction(nn.Module):  # return log likelihood of a Gaussian N(z | x, noise^2 * I)
    def __init__(self, c = torch.tensor(-2.0), opt = True):
        super(LikFunction, self).__init__()
        if type(c) != torch.Tensor:
            c = torch.tensor(c)
        self.noise = nn.Parameter(c, requires_grad = opt)

    def forward(self, o, x):  # both are n_sample, n_dim -- assert that they have the same dim -- for future use
        assert (len(o.size()) == 2) and (len(o.size()) == 2), \
            "Input matrices must be 2D"
        assert (o.size(0) == x.size(0)) and (o.size(1) == x.size(1)), \
            "Input matrices are supposed to be of the same size"
        diff = o - x
        n, d = o.size(0), o.size(1)
        output = -0.5 * n * d * torch.log(torch.tensor(2 * np.pi)) - 0.5 * n * d * self.noise \
                 -0.5 * torch.exp(-2.0 * self.noise) * torch.trace(torch.mm(diff.t(), diff))
        return output

