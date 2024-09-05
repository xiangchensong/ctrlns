import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy import optimize
from sklearn.metrics import accuracy_score
from .munkres import Munkres

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort

def compute_mcc(mus_train, ys_train, correlation_fn):
    """Computes score based on both training and testing codes and factors."""
    result = np.zeros(mus_train.shape)
    result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
    for i in range(len(mus_train) - len(ys_train)):
        result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
    corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
    mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
    return mcc
