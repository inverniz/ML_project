
"""some machine learning methods for project 1."""
import numpy as np
import numpy.linalg as la
from helpers import *

"""Linear regression using gradient descent."""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    # Arbitrary, needs tweaking
    prev_loss = 10000
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient
        loss = compute_mse(y, tx, w)
        if n_iter != 0 and np.abs(loss - prev_loss) < 1e-3:
            break
        prev_loss = loss
    return w, compute_mse(y, tx, w)


"""Linear regression using stochastic gradient descent."""
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, seed=1):
    np.random.seed(seed)
    data_size = len(y)
    w = initial_w
    for idx in np.random.randint(0,data_size-1,max_iters):
        minibatch_y = np.array([y[idx]])
        minibatch_tx = np.array([tx[idx]])
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)/data_size
        w = w - gamma * gradient
    return w, compute_mse(y, tx, w)


"""Least squares regression using normal equations."""
def least_squares(y, tx):
    w = la.pinv(tx) @ y
    return w, compute_mse(y, tx, w)


"""Ridge regression using normal equations."""
def ridge_regression(y, tx, lambda_):
    w = la.pinv(tx.T @ tx + (lambda_ * 2 * tx.shape[0]) * np.identity(tx.shape[1])) @ tx.T @ y
    return w, compute_mse(y, tx, w)


"""Logistic regression using gradient descent or SGD."""
def logistic_regression(y, tx, initial_w, max_iters, gamma, seed=1):
    np.random.seed(seed)
    w = initial_w
    data_size = len(y)
    for iter, idx in enumerate(np.random.randint(0,data_size-1,max_iters)):
        minibatch_y = np.array(y[idx])
        minibatch_tx = np.array([tx[idx]])
        grad = compute_gradient_logistic(minibatch_y, minibatch_tx, w)
        w = w - gamma * grad
    #return w, compute_mse(y, tx, w)
    return w, compute_loss_logistic(y, tx, w)


"""Regularized logistic regression using gradient descent or SGD."""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, seed=1):
    np.random.seed(seed)
    w = initial_w
    data_size = len(y)
    for iter, idx in enumerate(np.random.randint(0,data_size-1,max_iters)):
        minibatch_y = np.array(y[idx])
        minibatch_tx = np.array([tx[idx]])
        grad = compute_gradient_logistic_reg(minibatch_y, minibatch_tx, w, lambda_)
        w = w - gamma * grad
    #return w, compute_mse(y, tx, w)
    return w, compute_loss_logistic_reg(y, tx, w, lambda_)
