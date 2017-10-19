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

def cross_validation(y, tx, k_fold, function, loss_function, sup_args={}, sup_args_loss={}, seed = 1):
    k_indices = build_k_indices(y, k_fold, 1)
    total_loss_tr = 0
    total_loss_te = 0
    for k in range(k_fold):
        train_x = np.concatenate([tx[k_indices[i]] for i, idx in enumerate(k_indices) if i != k])
        train_y = np.concatenate([y[k_indices[i]] for i, idx in enumerate(k_indices) if i != k])
        test_x = tx[k_indices[k]]
        test_y = y[k_indices[k]]

        args = {'y': train_y, 'tx': train_x, **sup_args}
        w, loss_tr = function(**args)

        args_loss = {'y': test_y, 'tx': test_x, 'w': w, **sup_args_loss}
        loss_te = loss_function(**args_loss)

        total_loss_tr += loss_tr
        total_loss_te += loss_te

    return total_loss_tr/k_fold, total_loss_te/k_fold

