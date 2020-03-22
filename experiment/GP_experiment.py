from model.utility import *
from model.kernel import *
from approximation.ij_approximation import *
from model.gp import *
from data.data_loader import *
import copy


if __name__ == '__main__':

    #n_data, x_dim, k = 1000, 5, 100
    #data = generate_synthetic_data(n_data, x_dim)

    device = get_cuda_device()

    data = load_mauna_loa_data()
    n_data, x_dim, k = data['X_train'].shape[0], data['X_train'].shape[1], 100

    data['X_train'], data['Y_train'] = data['X_train'].to(device), data['Y_train'].to(device)
    data['X_test'], data['Y_test'] = data['X_test'].to(device), data['Y_test'].to(device)

    devs, Y_mean, Y_dev = compute_mean_dev(data['X_train'], data['Y_train'])
    devs, Y_mean, Y_dev = devs.to(device), Y_mean.to(device), Y_dev.to(device)
    mask = leave_out_indices(data['X_train'].shape[0], k)  # leave out the last K data points

    model = GP(x_dim, devs = devs, Y_mean = Y_mean, Y_dev = Y_dev)
    init_params = model.extract_params()
    MLE_params = optimize(copy.deepcopy(init_params), model, data['X_train'], data['Y_train'], n_iter = 1000, verbose = True)

    #pos_mean, _ = model.predict(data['X_test'], data['X_train'], data['Y_train'])
    #print(rmse(pos_mean, data['Y_test']))

    n = data['X_train'].shape[0]
    #lo_devs, lo_Y_mean, lo_Y_dev = compute_mean_dev(data['X_train'][0 : n - k, :], data['Y_train'][0 : n - k, :])
    #lo_init_params = create_params(lo_devs, mean = lo_Y_mean, sn = lo_Y_dev)
    no_ij_params = leave_out_optimize(copy.deepcopy(init_params), model, data['X_train'], data['Y_train'], mask,
                                      n_iter = 1000, verbose = False)
    ij_params = leave_out_optimize(copy.deepcopy(init_params), model, data['X_train'], data['Y_train'], mask,
                         do_IJ = True, MLE_params = copy.deepcopy(MLE_params))

    norm_difference = normalized_norm_difference(no_ij_params, ij_params)
    print(norm_difference)

