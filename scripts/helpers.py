import numpy as np

def compute_mse(y, tx, w):
    e = y.reshape(y.shape[0], -1) - tx.dot(w)
    print(e.shape)
    return (e.T.dot(e)/(2*y.shape[0]))[0][0]

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    return -tx.T.dot(e)/y.shape[0]

def grad_logistic(y, x, w):
    return x.dot(logistic_func(x.T.dot(w)) - y)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(np.exp(-t)+1)

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    sigma_xn_w = sigmoid(tx @ w)
    cat1_cost = y.T @ np.log(sigma_xn_w)
    cat2_cost = (1-y).T @ np.log(1-sigma_xn_w)
    cost = cat1_cost + cat2_cost
    return - cost

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)

