def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    return -tx.T.dot(e)/y.shape[0]
