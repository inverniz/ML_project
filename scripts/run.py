"""executing the machine learning methods"""
import numpy as np
from proj1_helpers import *
from implementations import *
  
def main():
    y, tx, ids = load_csv_data("data/train.csv", sub_sample=True)
    initial_w = np.zeros(30)
    max_iters = 10
    gamma = 0.7
    lambda_ = 5
    w1, loss1 = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    w2, loss2 = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
    w3, loss3 = least_squares(y, tx)
    w4, loss4 = ridge_regression(y, tx, lambda_)
    
    
    predictions1 = predict_labels(w1, tx)
    predictions2 = predict_labels(w2, tx)
    predictions3 = predict_labels(w3, tx)
    predictions4 = predict_labels(w4, tx)
    
    create_csv_submission(ids, predictions1, "predictions1.csv")
    create_csv_submission(ids, predictions1, "predictions2.csv")
    create_csv_submission(ids, predictions3, "predictions3.csv")
    create_csv_submission(ids, predictions4, "predictions4.csv")

if __name__== "__main__":
  main()
