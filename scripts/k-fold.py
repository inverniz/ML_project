import numpy as np
from helpers import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_fold, function, sup_args):
    k_indices = build_k_indices(y, k_fold, seed)
    total_loss_tr = 0
    total_loss_te = 0
    for k in range(k-fold):
        train_x = np.concatenate([x[k_indices[i]] for i, idx in enumerate(k_indices) if i != k])
        train_y = np.concatenate([y[k_indices[i]] for i, idx in enumerate(k_indices) if i != k])
        test_x = x[k_indices[k]]
        test_y = y[k_indices[k]]

        loss_tr, w = function(y, tx, *[sup_args])
        loss_te = compute_mse(y, tx, w)

        total_loss_tr += loss_tr
        total_loss_te += loss_te

    return total_loss_tr/k_fold, total_loss_te/k_fold

