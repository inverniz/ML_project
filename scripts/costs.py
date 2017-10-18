import numpy as np

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    return e.dot(e)/(2*y.shape[0])

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))
