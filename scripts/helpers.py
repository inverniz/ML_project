import numpy as np
import numpy.linalg as la
from scipy.misc import logsumexp
from implementations import *

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    return (e.T.dot(e)/(2*y.shape[0]))

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/y.shape[0]
    return grad #/ la.norm(grad)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(np.exp(-t)+1)

def build_poly(tx, degree):
    if degree <= 1:
        return tx
    return np.column_stack([tx] + [tx[:,1:]**k for k in range(2,degree+1)])

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
    grad = tx.T @ (sigmoid(tx @ w) - y)
    return grad# / la.norm(grad)

def compute_gradient_logistic_reg(y, tx, w, lambda_):
    return compute_gradient_logistic(y, tx, w) + lambda_ * w

def standardize(x):
    return (x - np.mean(x, axis=0))/np.std(x, axis=0)

def ridge_regression_with_poly(y, tx, lambda_, degree):
    tx_ridge = build_poly(tx, degree)
    return ridge_regression(y, tx_ridge, lambda_)

def compute_mse_with_poly(y, tx, w, degree):
    tx_poly = build_poly(tx, degree)
    return compute_mse(y, tx_poly, w)

def accuracy(y, tx, w):
    res = tx.dot(w)
    res[res <= 0] = -1
    res[res > 0]  = 1
    return 1-np.sum(y == res)/len(y)

def accuracy_with_poly(y, tx, w, degree):
    return accuracy(y, build_poly(tx, degree), w)
