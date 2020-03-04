import numpy as np

def IoU(y, pred):
    return np.count_nonzero(y & pred) / np.count_nonzero(y | pred)

def confusion_matrix(y, pred, threshold):
    pass
