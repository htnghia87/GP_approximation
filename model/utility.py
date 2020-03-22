import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalized_norm_difference(no_ij_params, ij_params):
    res = dict([])
    for key in ij_params.keys():
        a, b = ij_params[key], no_ij_params[key]
        res[key] = torch.norm(b - a) / torch.norm(b)
        res[key] = res[key].item()
    return res

def create_params(scales, mean = 1.0, sn = -0.0, noise = -2.0):
    params = dict([])
    params['scales'] = scales
    if type(mean) != torch.Tensor:
        mean = torch.tensor(mean)
    if type(sn) != torch.Tensor:
        sn = torch.tensor(sn)
    if type(noise) != torch.Tensor:
        noise = torch.tensor(noise)
    params['mean'] = mean
    params['signal'] = sn
    params['noise'] = noise
    return params

def leave_out_indices(n, k):
    mask = torch.ones((n,)).to(device)
    mask[n - k :] = -1.0
    return mask

def compute_mean_dev(X_train, Y_train):
    n_dim = X_train.shape[1]
    devs = torch.zeros(n_dim, 1)
    for i in range(n_dim) :
        devs[i, 0] = torch.log((2 ** 0.5) * torch.std(X_train[:, i]))
    Y_mean = torch.mean(Y_train[:, 0])
    Y_dev = torch.log((2 ** 0.5) * torch.std(Y_train[:, 0]))
    return devs, Y_mean.float(), Y_dev.float()

def generate_synthetic_data(n_data, n_dim):
    fix_randomizer()
    data = dict()
    A = 5. * ts(np.random.random((n_dim, 1)))
    data['X_train'] = ts(np.random.random((int(n_data * 0.9), n_dim)))
    data['X_test'] = ts(np.random.random((n_data - data['X_train'].shape[0], n_dim)))
    data['Y_train'] = torch.mm(data['X_train'], A)
    data['Y_test'] = torch.mm(data['X_test'], A)
    return data

def rmse(Y_pred, Y_true):  # both are of sizes sample by 1
    n = Y_pred.shape[0]
    diff = Y_pred - Y_true
    error = torch.sqrt(1.0 * torch.mm(diff.t(), diff) / n)
    return error

def ts(X):
    return torch.tensor(X)

def dt(X):
    return X.detach().numpy()

def get_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device

def compute_hessian(grad, params):  # obsolete
    grad = grad.reshape(-1)
    d = len(grad)
    H = torch.zeros((d, d))
    for i, g in enumerate(grad):
        res = torch.autograd.grad(g, params, create_graph = True)
        g2 = torch.cat([item.view(-1) for item in res])
        H[i] = g2
    return H

def compute_derivative(func, params, hessian = False):
    func.backward(retain_graph = True)
    grads = torch.autograd.grad(func, params, create_graph = True, retain_graph = True)  # get the gradient
    G = torch.cat([g.view(-1) for g in grads])  # reformat the gradient into a single vector
    H = None
    if hessian is True:  # if Hessian is needed
        d = G.shape[0]
        H = []
        for i in range(d):  # for each gradient component, compute its gradient
            res = torch.autograd.grad(G[i], params, retain_graph = True)
            res = torch.cat([item.view(-1) for item in res]).view(d, 1)
            H.append(res)
        H = torch.cat(H, dim = 1)  # reformat the second derivative into one single Hessian matrix
    return G, H

def fix_randomizer(np_seed = 1000, torch_seed = 20000):
    # for reproducibility
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)