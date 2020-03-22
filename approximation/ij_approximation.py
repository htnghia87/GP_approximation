from model.utility import *
from model.kernel import *
from model.gp import *

def optimize(init_params, model, X_train, Y_train, n_iter = 100, verbose = False):
    fix_randomizer()
    model.load_params(init_params)
    return model.optimize(X_train, Y_train, n_iter = n_iter, verbose = verbose)

def leave_out_optimize(init_params, model, X_train, Y_train, mask, do_IJ = False, MLE_params = None, n_iter = 100, verbose = False):
    fix_randomizer()
    in_pos = (mask > 0).flatten().nonzero().flatten()
    ou_pos = (mask < 0).flatten().nonzero().flatten()

    X_in = X_train[in_pos, :].view(-1, model.x_dim)
    Y_in = Y_train[in_pos, :].view(-1, 1)
    X_ou = X_train[ou_pos, :].view(-1, model.x_dim)
    Y_ou = Y_train[ou_pos, :].view(-1, 1)

    if do_IJ is False:
        model.load_params(init_params)
        leave_out_params = model.optimize(X_in, Y_in, n_iter = n_iter, verbose = verbose)
    else:
        assert MLE_params is not None, 'MLE params must be provided for infinitesimal jackknife (IJ) approximation'
        model.load_params(MLE_params)
        leave_out_params = model.IJ_optimize(X_in, Y_in, X_ou, Y_ou)

    return leave_out_params
