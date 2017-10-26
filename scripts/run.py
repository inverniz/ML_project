import numpy as np
from proj1_helpers import *
from implementations import *

y, x, ids = load_csv_data('../data/train.csv')
print("Shape of x:",x.shape)
print("Shape of y:",y.shape)
labels = np.array(np.genfromtxt('../data/train.csv', delimiter=",", names=True).dtype.names[2:])
filter_ = [idx for idx, label in enumerate(labels) if not 'phi' in label]
x = x[:, filter_]
x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
y = (y+1)/2.0
num_samples = len(y)
tx = np.c_[np.ones(num_samples), x]
np.random.seed(1)
#w, loss = logistic_regression(y, tx, np.zeros(tx.shape[1]), 10000000, 0.0081491274690207397)
w, loss = ridge_regression(y, tx, 0)
#w, loss = least_squares_GD(y, tx, np.zeros(tx.shape[1]), 100000, 0.0001)
#w, loss = reg_logistic_regression(y, tx, 2.15, np.zeros(tx.shape[1]), 100000, 1e-5)
print(loss)
#y_test, x_test, ids_test = load_csv_data('../data/test.csv')
#x_test = x_test[:, filter_]
#x_test = (x_test - np.mean(x, axis=0))/np.std(x, axis=0)
#tx_test = np.c_[np.ones(len(y_test)),x_test]
#y_pred = predict_labels(w, tx_test)
#create_csv_submission(ids_test, y_pred, "test.csv")
