import numpy as np
import numpy.linalg as la
from proj1_helpers import *

### Loss Functions ###

def compute_mse(y, tx, w):
    """Compute the mean squared error"""
    e = y - tx.dot(w)
    return (e.T.dot(e)/(2*y.shape[0]))

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    sigma_xn_w = sigmoid(tx @ w)
    loss = np.sum(y.T @ np.log(sigma_xn_w) + (1-y).T @ np.log(1-sigma_xn_w))
    return - loss/y.shape[0]

def compute_loss_logistic_reg(y, tx, w, lambda_):
    """"Compute the cost for the logisitc mehtods"""
    return compute_loss_logistic(y, tx, w) + lambda_/2.0 * (w.T @ w)

### Gradient Functions ###

def compute_gradient(y, tx, w):
    """Compute the gradient for the point w"""
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/y.shape[0]
    return grad

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(np.exp(-t)+1)

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    grad = tx.T @ (sigmoid(tx @ w) - y)
    return grad

def compute_gradient_logistic_reg(y, tx, w, lambda_):
    """Compute the logistic gradient and adds penalty term"""
    return compute_gradient_logistic(y, tx, w) + lambda_ * w

### Feature Processing ###
def build_poly(tx, degree):
    """Construct a polynomial basis of degree"""
    if degree <= 1:
        return tx
    return np.column_stack([tx] + [tx[:,1:]**k for k in range(2,degree+1)])

def cut_at_percentile(x, percentile):
    """Cut x at a certain percentile"""
    res = x.copy()
    for i in range(x.shape[1]):
        max_val = np.percentile(x[:,i], percentile, interpolation='midpoint')
        res[x[:, i] > max_val, i] = max_val
    return res

def get_group(x, y, n):
    """Get group number n, cf. documentation for details"""

    # Get the numbers of jet and the presence of no of mass
    num_jet = int(n/2)
    mass = n % 2 == 0

    # We will keep only values with a certain number of jet
    mask = x[:, 22] == num_jet
    # if we are asked for 2 jet, we group with 3 jet
    if num_jet == 2:
        mask = mask | (x[:, 22] == 3)

    # Add mass or not to selection
    if mass:
        mask = mask & (x[:, 0] != -999)
    else:
        mask = mask & (x[:, 0] == -999)
    # Mask the rows
    x_group = x[mask]

    # Remove null columns
    group_mean = np.mean(x_group, axis=0)
    x_group = x_group[:, (group_mean != -999) & (group_mean != 0) & (group_mean != num_jet)]

    # Mask the row of the results
    y_group = y[mask]
    return x_group, y_group

### General helpers ###

def ridge_regression_with_poly(y, tx, lambda_, degree):
    """Wrapper to run ridge regression with a polynomial basis"""
    from implementations import ridge_regression
    tx_ridge = build_poly(tx, degree)
    return ridge_regression(y, tx_ridge, lambda_)

def accuracy(y, tx, w, is_sigmoid=False):
    """Reports the fraction of errors"""
    res = tx.dot(w)

    # Take into account if the function is using the sigmoid
    if is_sigmoid:
        res = sigmoid(res)
        res[res <= 0.5] = 0
        res[res > 0.5] = 1
    else:
        res[res <= 0] = -1
        res[res > 0]  = 1
    return 1-np.sum(y == res)/len(y)

def accuracy_with_poly(y, tx, w, degree, is_sigmoid=False):
    """Reports the fraction of errors, with a polynomial basis"""
    return accuracy(y, build_poly(tx, degree), w, is_sigmoid)

def run_for_group(x, y, n, function, sup_args = {}):
    """Run the given function for a given group"""

    x_group, y_group = get_group(x, y, n)
    mean = np.mean(x_group, axis=0)
    std = np.std(x_group, axis=0)
    x_group = (x_group - mean)/std
    tx_group = np.c_[np.ones(len(y_group)), x_group]

    # We unwrap the sup args and wrap them again to call the function
    args = {'y': y_group, 'tx': tx_group, **sup_args}
    w, loss = function(**args)
    return w, mean, std

def predict_for_group(x, mean, std, w, n, num_pred, sup_args):
    """Predict the labels for a given group"""
    # We get the group and the corresponding indices
    x_group, idxs = get_group(x, np.arange(num_pred), n)
    x_group = (x_group - mean)/std
    tx_group = np.c_[np.ones(len(idxs)), x_group]
    if 'degree' in sup_args:
        tx_group = build_poly(tx_group, sup_args['degree'])
    y_pred = predict_labels(w, tx_group)
    return y_pred, idxs

def run_and_predict(x_train, y_train, x_test, function, sup_args=[{},{},{},{},{},{}]):
    """Run for all groups and predict the final output"""
    num_pred_train = x_train.shape[0]
    num_pred_test = x_test.shape[0]
    y_pred_train = np.zeros(num_pred_train)
    y_pred_test = np.zeros(num_pred_test)
    for n in range(6):
        w, mean, std = run_for_group(x_train, y_train, n, function, sup_args[n])
        y_pred_group_train, idxs = predict_for_group(x_train, mean, std, w, n, num_pred_train, sup_args[n])
        # We put the predictions at the rights indices
        np.put(y_pred_train, idxs, y_pred_group_train)
        y_pred_group_test, idxs = predict_for_group(x_test, mean, std, w, n, num_pred_test, sup_args[n])
        np.put(y_pred_test, idxs, y_pred_group_test)
    return y_pred_train, y_pred_test
