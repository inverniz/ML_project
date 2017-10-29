import numpy as np
import numpy.linalg as la
from scipy.misc import logsumexp
from proj1_helpers import *

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    return (e.T.dot(e)/(2*y.shape[0]))

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/y.shape[0]
    return grad / la.norm(grad)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(np.exp(-t)+1)

def build_poly(tx, degree):
    if degree <= 1:
        return tx
    return np.column_stack([tx] + [tx[:,1:]**k for k in range(2,degree+1)])

def logistic_GD(y, tx, initial_w,lambda_, max_iters, gamma, degree):
    tx = build_poly(tx, degree)
    w = np.zeros(tx.shape[1])
    prev_loss = 10000
    minibatches_y, minibatches_tx = get_minibatches(y, tx, 4096)
    #prev_grads = w
    velocity = w
    eta = 0.9
    for n_iter in range(max_iters):
        idx_batch = np.random.randint(0, len(minibatches_y))
        minibatch_y = minibatches_y[idx_batch]
        minibatch_tx = minibatches_tx[idx_batch]

        # ADAgrad
        #prev_grads += grad**2
        #w = w - gamma * np.diag(1.0/(np.sqrt(prev_grads)+1e-9)) @ grad

        # Vanilla
        #grad = compute_gradient_logistic_reg(minibatch_y, minibatch_tx, w, lambda_)
        #w = w - gamma * grad

        # Nesterov
        grad = compute_gradient_logistic_reg(minibatch_y, minibatch_tx, w - eta*velocity, lambda_)
        velocity = eta*velocity + gamma * grad
        w = w - velocity
    
        loss = compute_loss_logistic_reg(minibatch_y, minibatch_tx, w, lambda_)
        #if n_iter != 0 and np.abs(loss - prev_loss) < 1e-6:
        #    break
        #if (n_iter+1) % 100 == 0:
        #    print(n_iter, np.abs(loss - prev_loss))
        prev_loss = loss
    return w, loss


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

def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]

def get_minibatches(y, tx, batch_size):
    minibatches_y = []
    minibatches_tx = []
    data_size = len(y)
    for i in range(0, data_size, batch_size):
        minibatches_y.append(y[i:i+batch_size])
        minibatches_tx.append(tx[i:i+batch_size])
    return minibatches_y, minibatches_tx

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    sigm_xn_w = sigmoid(tx @ w)
    return tx.T @ (sigm_xn_w * (1-sigm_xn_w) * np.identity(y.shape[0])) @ tx

def logistic_regression_newton(y, tx, lambda_, initial_w, max_iters):
    w = initial_w
    #prev_loss = 1000
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 16, max_iters):
        #minibatch_y, minibatch_tx = sample_data(y, tx, 1, 16)
        loss = compute_loss_logistic(minibatch_y, minibatch_tx, w)
        w = w - lambda_ * (la.pinv(calculate_hessian(minibatch_y, minibatch_tx, w)) @ compute_gradient_logistic(minibatch_y, minibatch_tx, w))
        #if np.abs(loss - prev_loss) < treshold:
        #    break
        #prev_loss = loss
    return w, loss

def logistic_regression_newton_with_poly(y, tx, lambda_, initial_w, max_iters, degree):
    tx_poly = build_poly(tx, degree)
    return logistic_regression_newton(y, tx_poly, lambda_, np.zeros(tx_poly.shape[1]), max_iters)


def compute_gradient_logistic_reg(y, tx, w, lambda_):
    return compute_gradient_logistic(y, tx, w) + lambda_ * w

def standardize(x):
    return (x - np.mean(x, axis=0))/np.std(x, axis=0)

def least_squares_GD_with_poly(y, tx, initial_w, max_iters, gamma, degree):
    from implementations import least_squares_GD
    tx_augmented = build_poly(tx, degree)
    return least_squares_GD(y, tx_augmented, np.zeros(tx_augmented.shape[1]), max_iters, gamma)

def ridge_regression_with_poly(y, tx, lambda_, degree):
    from implementations import ridge_regression
    tx_ridge = build_poly(tx, degree)
    return ridge_regression(y, tx_ridge, lambda_)

def reg_logistic_reg_with_poly(y, tx, lambda_, initial_w, max_iters, gamma, degree,seed=1):
    from implementations import reg_logistic_regression
    tx_augmented = build_poly(tx, degree)
    return reg_logistic_regression(y, tx_augmented, lambda_, np.zeros(tx_augmented.shape[1]), max_iters, gamma)

def compute_mse_with_poly(y, tx, w, degree):
    tx_poly = build_poly(tx, degree)
    return compute_mse(y, tx_poly, w)

def cut_at_percentile(x, percentile):
    res = x.copy()
    for i in range(x.shape[1]):
        max_val = np.percentile(x[:,i], percentile, interpolation='midpoint')
        res[x[:, i] > max_val, i] = max_val
    return res


def accuracy(y, tx, w, is_sigmoid=False):
    res = tx.dot(w)
    if is_sigmoid:
        res = sigmoid(res)
        res[res <= 0.5] = 0
        res[res > 0.5] = 1
    else:
        res[res <= 0] = -1
        res[res > 0]  = 1
    return 1-np.sum(y == res)/len(y)


def accuracy_with_poly(y, tx, w, degree, is_sigmoid=False):
    return accuracy(y, build_poly(tx, degree), w, is_sigmoid)

def get_group(x, y, n):
    num_jet = int(n/2)
    mass = n % 2 == 0

    mask = x[:, 22] == num_jet
    if num_jet == 2:
        mask = mask | (x[:, 22] == 3)
    if mass:
        mask = mask & (x[:, 0] != -999)
    else:
        mask = mask & (x[:, 0] == -999)
    x_group = x[mask]
    group_mean = np.mean(x_group, axis=0)
    x_group = x_group[:, (group_mean != -999) & (group_mean != 0) & (group_mean != num_jet)]
    y_group = y[mask]
    return x_group, y_group

def run_for_group(x, y, n, function, sup_args = {}):
    x_group, y_group = get_group(x, y, n)
    mean = np.mean(x_group, axis=0)
    std = np.std(x_group, axis=0)
    x_group = (x_group - mean)/std
    tx_group = np.c_[np.ones(len(y_group)), x_group]

    args = {'y': y_group, 'tx': tx_group, **sup_args}
    w, loss = function(**args)
    return w, mean, std

def predict_for_group(x, mean, std, w, n, num_pred, sup_args):
    x_group, idxs = get_group(x, np.arange(num_pred), n)
    x_group = (x_group - mean)/std
    tx_group = np.c_[np.ones(len(idxs)), x_group]
    if 'degree' in sup_args:
        tx_group = build_poly(tx_group, sup_args['degree'])
    y_pred = predict_labels(w, tx_group)
    return y_pred, idxs

def run_and_predict(x_train, y_train, x_test, function, sup_args=[{},{},{},{},{},{}]):
    num_pred_train = x_train.shape[0]
    num_pred_test = x_test.shape[0]
    y_pred_train = np.zeros(num_pred_train)
    y_pred_test = np.zeros(num_pred_test)
    for n in range(6):
        w, mean, std = run_for_group(x_train, y_train, n, function, sup_args[n])
        y_pred_group_train, idxs = predict_for_group(x_train, mean, std, w, n, num_pred_train, sup_args[n])
        np.put(y_pred_train, idxs, y_pred_group_train)
        y_pred_group_test, idxs = predict_for_group(x_test, mean, std, w, n, num_pred_test, sup_args[n])
        np.put(y_pred_test, idxs, y_pred_group_test)
    return y_pred_train, y_pred_test
