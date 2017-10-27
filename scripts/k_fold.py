import numpy as np
from helpers import standardize, accuracy

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_fold, function, loss_function, sup_args={}, sup_args_loss={}, seed = 1):
    k_indices = build_k_indices(y, k_fold, 1)
    total_loss_tr = 0
    total_loss_te = 0
    for k in range(k_fold):
        train_x = np.concatenate([x[k_indices[i]] for i, idx in enumerate(k_indices) if i != k])
        train_mean, train_std = np.mean(train_x, axis=0), np.std(train_x, axis=0)
        train_x = (train_x - train_mean)/train_std
        train_y = np.concatenate([y[k_indices[i]] for i, idx in enumerate(k_indices) if i != k])
        train_tx = np.c_[np.ones(train_y.shape[0]), train_x]

        test_x = x[k_indices[k]]
        test_x = (test_x - train_mean)/train_std
        test_y = y[k_indices[k]]
        test_tx = np.c_[np.ones(test_y.shape[0]), test_x]

        args = {'y': train_y, 'tx': train_tx, **sup_args}
        w, loss_tr = function(**args)
        args_loss = {'y': train_y, 'tx': train_tx, 'w': w, **sup_args_loss}
        loss_tr = loss_function(**args_loss)

        args_loss = {'y': test_y, 'tx': test_tx, 'w': w, **sup_args_loss}
        loss_te = loss_function(**args_loss)

        total_loss_tr += loss_tr
        total_loss_te += loss_te

    return total_loss_tr/k_fold, total_loss_te/k_fold

