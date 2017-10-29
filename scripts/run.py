import numpy as np
from proj1_helpers import *
from implementations import *
from helpers import *

# These parameters for the ridge regression were found using a 10-fold cross validation
args = [{'lambda_': 0, 'degree': 7}, {'lambda_': 0, 'degree': 5}, {'lambda_': 1e-4, 'degree': 9},\
       {'lambda_': 1.66e-8, 'degree': 4}, {'lambda_': 4.64e-4, 'degree': 8}, {'lambda_': 0, 'degree': 4}]


y, x, ids = load_csv_data('../data/train.csv')
y_test, x_test, ids_test = load_csv_data('../data/test.csv')

# We remove outliers above 95th percentile
x = cut_at_percentile(x, 95)
x_test = cut_at_percentile(x_test, 95)

y_pred_train, y_pred_test = run_and_predict(x, y, x_test, ridge_regression_with_poly, args)

print(1-np.sum(y_pred_train==y)/len(y))
create_csv_submission(ids_test, y_pred_test, "submission.csv")
