"""some machine learning methods for project 1."""
import numpy as np

"""Compute the gradient for gradient descent."""
def compute_gradient_GD(y, tx, w):
	error = y - tx.dot(w)
	n = y.shape[0]
	x_transposed = tx.T
	gradient = -x_transposed.dot(error)/n
	
	return gradient
	
"""Compute the gradient for stochastic gradient descent."""
def compute_gradient_SGD(yn, txn, w):
	error = yn - txn.dot(w)
	xn_transposed = txn.T
	gradient = -xn_transposed.dot(error)
	
	return gradient

"""Calculate the loss for GD."""
def compute_loss(y, tx, w):
    error = y - tx.dot(w)
    n = y.shape[0]
    loss = error.T.dot(error)/(2*n)
    
    return loss
       
"""Linear regression using gradient descent."""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_GD(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma*gradient
    return w, loss

"""Linear regression using stochastic gradient descent."""
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    n = y.shape[0]
    for n_iter in range(max_iters):
	    index = np.random.random_integers(0, n-1)
	    yn = y[index]
	    txn = tx[index]
	    gradient = compute_gradient_SGD(yn, txn, w)
	    w = w - gamma*gradient	
    loss = compute_loss(y, tx, w)
    return w, loss
	
"""Least squares regression using normal equations."""
def least_squares(y, tx):
    w = np.linalg.lstsq(tx, y)[0]
    loss = compute_loss(y, tx, w)
    return w, loss


"""Ridge regression using normal equations."""
def ridge_regression(y, tx, lambda_ ):
    n = tx.shape[0]
    dimensionality = tx.shape[1]
    txt = tx.T
    id_matrix = np.eye(dimensionality)
    _lambda_ = 2*n*lambda_
    pseudoinverse = np.linalg.pinv(txt.dot(tx) + _lambda_*id_matrix)
    
    w = pseudoinverse.dot(txt).dot(y)
    loss = compute_loss(y, tx, w)
    return w, loss


"""Logistic regression using gradient descent or SGD."""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = 0
    loss = 0
    
    return w, loss


"""Regularized logistic regression using gradient descent or SGD."""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = 0
    loss = 0
    
    return w, loss
