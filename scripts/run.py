import numpy as np
from proj1_helpers import *
from implementations import *

y, x, ids = load_csv_data('../data/train.csv')
print("Shape of x:",x.shape)
print("Shape of y:",y.shape)
x = (x - np.mean(x))/np.std(x)
num_samples = len(y)
tx = np.c_[np.ones(num_samples), x]
w, loss = logistic_regression(y, tx, np.zeros((tx.shape[1],1)), 1000000,1e-7)
print(loss)
y_test, x_test, ids_test = load_csv_data('../data/test.csv')
x_test = (x_test - np.mean(x_test))/np.std(x_test)
tx_test = np.c_[np.ones(len(y_test)),x_test]
y_pred = predict_labels(w, tx_test)
create_csv_submission(ids_test, y_pred, "test.csv")
