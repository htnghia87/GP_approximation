import torch
import numpy as np
from sklearn import datasets
from model.utility import *

def load_mauna_loa_atmospheric_co2():
    ml_data = datasets.fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs

def load_mauna_loa_data():
    np.random.seed(2603)
    x, y = load_mauna_loa_atmospheric_co2()
    #x = np.split(x, [421, 471, 521])
    #y = np.split(y, [421, 471, 521])

    x = np.split(x, [471, 521])
    y = np.split(y, [471, 521])

    data = dict([])
    #data['dim'] = x[0].shape[1]
    data['X_train'] = ts(x[0]).float()
    data['X_test'] = ts(x[1]).float()
    #data['X_val'] = x[2]
    data['Y_train'] = ts(y[0].reshape(-1, 1)).float()
    data['Y_test'] = ts(y[1].reshape(-1, 1)).float()
    #data['X_val'] = y[2].reshape(-1, 1)

    return data
