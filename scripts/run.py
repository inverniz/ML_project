import numpy as np
from proj1_helpers import *
from implementations import *
from helpers import *

y, x, ids = load_csv_data('../data/train.csv')
print("Shape of x:",x.shape)
print("Shape of y:",y.shape)
labels = np.array(np.genfromtxt('../data/train.csv', delimiter=",", names=True).dtype.names[2:])
#filter_ = [idx for idx, label in enumerate(labels) if not 'phi' in label]
#x = x[:, filter_]
def x_y_for_jet(x, y, n, mass=True):
    jet_num = x[:, 22] == n
    if n == 2:
        jet_num = jet_num | (x[:, 22] == 3)
    if mass:
        jet_num = jet_num & (x[:, 0] != -999)
    else:
        jet_num = jet_num & (x[:, 0] == -999)
    x_jet = x[jet_num]
    y_jet = y[jet_num]
    jet_mean = np.mean(x_jet, axis=0)
    x_jet = x_jet[:, (jet_mean != -999) & (jet_mean != 0) & (jet_mean != n)]
    return x_jet, y_jet
ws = []
means = []
stds = []
losses = []
for n in range(3):
    for mass in [True, False]:
        x_jet, y_jet = x_y_for_jet(x, y, n, mass)
        mean = np.mean(x_jet, axis=0)
        means.append(mean)
        std = np.std(x_jet, axis=0)
        stds.append(std)
        x_jet = (x_jet-mean)/std
        tx_jet = np.c_[np.ones(len(y_jet)), x_jet]
        #tx_jet = build_poly(np.c_[np.ones(len(y_jet)), x_jet], 6)
        w, loss = ridge_regression(y_jet, tx_jet, 1e-10)
        ws.append(w)
        losses.append(accuracy(y_jet, tx_jet, w))
print(np.mean(losses))

y_test, x_test, ids_test = load_csv_data('../data/test.csv')
print('Preds loaded')
y_pred = np.zeros(len(ids_test))
for n in range(3):
    for mass in [True, False]:
        x_jet, idxs = x_y_for_jet(x_test,np.arange(len(ids_test)),n, mass)
        idx = 2*n + (int(mass)*-1+1)
        print(idx)
        x_jet = (x_jet - means[idx])/stds[idx]
        tx_jet = np.c_[np.ones(len(idxs)), x_jet]
        #tx_jet = build_poly(np.c_[np.ones(len(idxs)), x_jet], 6)
        y_jet = predict_labels(ws[idx], tx_jet)
        np.put(y_pred, idxs, y_jet)
create_csv_submission(ids_test, y_pred, "test.csv")
#x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
#y = (y+1)/2.0
#num_samples = len(y)
#tx = np.c_[np.ones(num_samples), x]
#np.random.seed(1)
##w, loss = logistic_regression(y, tx, np.zeros(tx.shape[1]), 10000000, 0.0081491274690207397)
#w, loss = ridge_regression(y, tx, 0)
#w, loss = least_squares_GD(y, tx, np.zeros(tx.shape[1]), 100000, 0.0001)
#w, loss = reg_logistic_regression(y, tx, 2.15, np.zeros(tx.shape[1]), 100000, 1e-5)
print(loss)
#y_test, x_test, ids_test = load_csv_data('../data/test.csv')
#x_test = x_test[:, filter_]
#x_test = (x_test - np.mean(x, axis=0))/np.std(x, axis=0)
#tx_test = np.c_[np.ones(len(y_test)),x_test]
#y_pred = predict_labels(w, tx_test)
#create_csv_submission(ids_test, y_pred, "test.csv")
