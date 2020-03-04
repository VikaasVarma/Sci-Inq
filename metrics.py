import numpy as np

def IoU(ys, preds):
    intersection = union = 0
    for i in range(len(ys)):
        intersection += np.count_nonzero(ys[i] | preds[i] == 0)
        union += np.count_nonzero(ys[i] | preds[i])
    return intersection / union

def confusion_matrix(ys, preds, threshold):
    pass
