"""executing the machine learning methods"""
import numpy as np
from proj1_helpers import *
from implementations import *
  
def main():
    y, tx, ids = load_csv_data("data/train.csv")
    initial_w = np.ones(30)
    max_iters = 10
    gamma = 1
    
    w1, loss1 = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    w2, loss2 = least_squares_SGD(y, tx, initial_w, max_iters, gamma)

    predictions1 = predict_labels(w1, tx)
    predictions2 = predict_labels(w2, tx)

    create_csv_submission(ids, predictions1, "predictions1.csv")
    create_csv_submission(ids, predictions1, "predictions2.csv")

if __name__== "__main__":
  main()
