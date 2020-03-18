import numpy as np

def IoU(ys, preds):
    intersection = union = 0
    for i in range(len(ys)):
        intersection += np.count_nonzero(ys[i] | preds[i] == 0)
        union += np.count_nonzero(ys[i] | preds[i])
    return intersection / union

def confusion_matrix(ys, preds):
    cmatrix = np.zeros(2,2)
    for i in range(len(ys)):
        cmatrix[0, 0] += np.count_nonzero(np.logical_not(ys[i] - preds[i]) & preds[i]) #true positive
        cmatrix[0, 1] += np.count_nonzero(np.logical_not(ys[i]) & preds[i]) #false positive
        cmatrix[1, 0] += np.count_nonzero(np.logical_not(preds[i]) & ys[i]) #false negative
        cmatrix[1, 1] += np.count_zero(ys[i] & preds[i])    #true negative
    return cmatrix/cmatrix.sum()

#subset false positive; counts number of mislabels excluding 0's
def mislabelled(ys, preds):
    miss = total = 0
    for i in range(len(ys)):
        miss += np.count_nonzero((ys[i] & preds[i]) & (ys[i] - preds[i]))
        total += ys[i].size
    return miss, total
