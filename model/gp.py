from model.kernel import *
from model.utility import *
import torch.optim as opt
import copy


class GP(nn.Module):
    def __init__(self, x_dim, devs = None, Y_mean = None, Y_dev = None):
        super(GP, self).__init__()
        self.x_dim = x_dim
        if (devs is None) and (Y_dev is None):
            self.cov = CovFunction(self.x_dim)
        if (devs is None) and (Y_dev is not None):
            self.cov = CovFunction(self.x_dim, c = Y_dev)
        if (devs is not None) and (Y_dev is None):
            self.cov = CovFunction(self.x_dim, scales = devs)
        if (devs is not None) and (Y_dev is not None):
            self.cov = CovFunction(self.x_dim, scales = devs, c = Y_dev)
        if Y_mean is None:
            self.mean = MeanFunction(opt = True)
        else:
            self.mean = MeanFunction(c = Y_mean, opt = True)
        self.lik = LikFunction(opt = True)
        self.NLL_hess = None  # supposed to be cached with the Hessian of the NLL over full training set

    def NLL(self, X_train, Y_train):  # X_train (_, x_dim) -- Y_train (_, 1)
        K = self.cov(X_train) + torch.exp(self.lik.noise * 2) * torch.eye(X_train.shape[0]).to(device)
        K_inv = torch.inverse(K)
        z = self.mean(X_train)
        u = z - Y_train
        output = 0.5 * X_train.shape[0] * np.log(2 * np.pi) + 0.5 * torch.logdet(K) \
                 + 0.5 * torch.mm(u.t(), torch.mm(K_inv, u))  # we need to minimize this
        return output

    def predict(self, X_test, X_train, Y_train):
        k_test = self.cov(X_test, X_train)
        K = self.cov(X_train) + torch.exp(self.lik.noise * 2) * torch.eye(X_train.shape[0]).to(device)
        K_inv = torch.inverse(K)
        pos_var = self.cov(X_test, X_test) - torch.mm(k_test, torch.mm(K_inv, k_test.t()))
        pos_mean = self.mean(X_test) + torch.mm(k_test, torch.mm(K_inv, Y_train - self.mean(X_train)))
        return pos_mean, pos_var

    def predictive_NLL(self, X_test, Y_test, X_train, Y_train):
        pos_mean, pos_var = self.predict(X_test, X_train, Y_train)
        pos_var = pos_var + torch.eye(X_test.shape[0]).to(device) * torch.exp(2.0 * self.lik.noise)
        pos_var_inv = torch.inverse(pos_var)
        n = Y_test.shape[0]
        z = pos_mean - Y_test
        res = 0.5 * n * np.log(2 * np.pi) + 0.5 * torch.logdet(pos_var) + \
              0.5 * torch.mm(z.t(), torch.mm(pos_var_inv, z))
        return res

    def extract_params(self):
        params = dict([])
        params['mean'] = copy.deepcopy(self.mean.mean.data.view(-1))
        params['scales'] = copy.deepcopy(self.cov.weights.data.view(-1))
        params['noise'] = copy.deepcopy(self.lik.noise.data.view(-1))
        params['signal'] = copy.deepcopy(self.cov.sn.data.view(-1))
        return params

    def load_params(self, params):
        self.cov.weights = nn.Parameter(params['scales'].view(self.x_dim, 1), requires_grad = True)
        self.cov.sn = nn.Parameter(params['signal'], requires_grad = True)
        self.mean.mean = nn.Parameter(params['mean'], requires_grad = True)
        self.lik.noise = nn.Parameter(params['noise'], requires_grad = True)

    @staticmethod
    def debug(model):
        for param in model.parameters():  # for debugging purpose: make sure all parameters are accounted for
            if param.requires_grad:
                print(param.data)

    @staticmethod
    def grad_param(model):
        res = []
        for param in model.parameters():  # for debugging purpose: make sure all parameters are accounted for
            if param.requires_grad:
                res.append(param)
        return res

    def cache_NLL_hess(self, X_train, Y_train):
        model = nn.ModuleList([self.cov, self.mean, self.lik])
        func = self.NLL(X_train, Y_train)
        var = list(model.parameters())
        func.backward(retain_graph = True)
        _, NLL_hess = compute_derivative(func, var, hessian = True)
        self.NLL_hess = NLL_hess

    def optimize(self, X_train, Y_train, n_iter = 100, verbose = False):
        model = nn.ModuleList([self.cov, self.mean, self.lik])
        optimizer = opt.Adam(model.parameters())
        for i in range(n_iter + 1):
            model.train()
            optimizer.zero_grad()
            GP_loss = self.NLL(X_train, Y_train)
            print('Training Iteration ' + str(i) + ' : ', GP_loss.item())
            if i < n_iter + 1:
                if verbose is True:
                    print(self.extract_params())
                    #self.debug(model)
                    GP_loss.backward(retain_graph = True)
                    var = self.grad_param(model)
                    grads = torch.autograd.grad(GP_loss, var, create_graph = True, retain_graph = True)
                    G = torch.cat([g.view(-1) for g in grads])
                    #print(G / torch.norm(G))
                    print(G)
                else:
                    GP_loss.backward()
                optimizer.step()
        print('Done')
        self.cache_NLL_hess(X_train, Y_train)
        opt_params = self.extract_params()
        return opt_params

    def IJ_parameter(self, var_ij):
        ij_params = dict([])
        ij_params['mean'] = var_ij[self.x_dim + 1].data.view(1)
        ij_params['scales'] = var_ij[0 : self.x_dim].data
        ij_params['noise'] = var_ij[-1].data.view(1)
        ij_params['signal'] = var_ij[self.x_dim].data.view(1)
        return ij_params

    def IJ_optimize(self, X_in, Y_in, X_ou, Y_ou):  # assume the current parameters are the MLE
        assert self.NLL_hess is not None, "MLE solution has not been cached for the IJ approximation"
        model = nn.ModuleList([self.cov, self.mean, self.lik])
        #model = nn.ModuleList([self.cov])
        model.train()
        pred_NLL = self.predictive_NLL(X_ou, Y_ou, X_in, Y_in)
        var = list(model.parameters())
        grad, hess = compute_derivative(pred_NLL, var, hessian = True)
        var_opt = torch.cat([var[i].data.view(-1) for i in range(len(var))])
        var_ij = var_opt + torch.mm(grad.view(1, -1), torch.inverse(self.NLL_hess - hess)).view(-1)
        #var_ij[self.x_dim + 1] = var_opt[self.x_dim + 1] # no approximation for mean
        #var_ij[-1] = var_opt[-1] # no approximation for noise
        #var_ij[0 : self.x_dim] = var_opt[0 : self.x_dim]  # no approximation for scales
        return self.IJ_parameter(var_ij)




