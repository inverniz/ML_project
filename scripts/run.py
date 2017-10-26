"""executing the machine learning methods"""
import numpy as np
from proj1_helpers import *
from implementations import *
  
def clean_and_standardize(tx):
    x = tx.copy()
    dimensionality = x.shape[1]
    for feature in range(dimensionality):
        feature_column = x[:, feature]
        valid_index = (feature_column != -999)
        feature_mean = np.mean(feature_column[valid_index])
        feature_column[~valid_index] = feature_mean
        feature_std = np.std(feature_column)
        feature_column = feature-feature_mean/feature_std
        
    return x

def remove_uniform_distribution(tx):
    x = np.delete(tx, [15, 18, 20, 25, 28], 1)
    return x

def get_index(x, value):
    # we want to consider feature 22, which is now feature 19 
    # since 3 features have been deleted.
    index = (x[:, 19] == value)
    return index

def divide_into_groups(x, y):
    #compute index for various groups
    group0 = get_index(x, 0)
    group1 = get_index(x, 1)
    group2a = get_index(x, 2)
    group2b = get_index(x, 3)
    group2 = np.logical_or(group2a, group2b)
    
    x_group0 = x[group0]
    x_group1 = x[group1]
    x_group2 = x[group2]
    
    y_group0 = y[group0]
    y_group1 = y[group1]
    y_group2 = y[group2]
    
    return x_group0,x_group1,x_group2,y_group0,y_group1,y_group2

def main():
    y, tx, ids = load_csv_data("data/train.csv", sub_sample=True)
    tx = clean_and_standardize(tx)
    tx = remove_uniform_distribution(tx)
    tx_0, tx_1, tx_2, y_0, y_1, y_2 = divide_into_groups(tx, y)
    
    #TODO: treat three groups separately
    print(y_0)
    
    #predictions = predict_labels(w, tx)
    #create_csv_submission(ids, predictions, "predictions1.csv")

if __name__== "__main__":
  main()
