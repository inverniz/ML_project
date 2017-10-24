import numpy as np
from scipy.misc import logsumexp

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    return (e.T.dot(e)/(2*y.shape[0]))

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

def log_sum_exp(x):
    return np.log(np.sum(b))
    
def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    sigma_xn_w = sigmoid(tx @ w)
    loss = 0
    #for i in range(y.shape[0]):
    #    sigma_xi_w = sigma_xn_w[i][0]
    #    loss += y[i] * np.log(sigma_xi_w) + (1-y[i])*np.log(1- sigma_xi_w)
        #loss += y[i] * -logsumexp(0.0,-txw[i][0]) - (1-y[i])*logsumexp(0.0,txw[i][0])
    loss = np.sum(y.T @ np.log(sigma_xn_w) + (1-y).T @ np.log(1-sigma_xn_w))
        #cat1_cost = y.T @ np.log(sigma_xn_w)
    #cat2_cost = (1-y).T @ np.log(1-sigma_xn_w)
    #cost = cat1_cost + cat2_cost
    return - loss/y.shape[0]

def compute_loss_logistic_reg(y, tx, w, lambda_):
    return compute_loss_logistic(y, tx, w) + lambda_/2.0 * (w.T @ w)

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)

def compute_gradient_logistic_reg(y, tx, w, lambda_):
    return compute_gradient_logistic(y, tx, w) + lambda_ * w
