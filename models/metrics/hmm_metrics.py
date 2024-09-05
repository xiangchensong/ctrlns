import numpy as np
from itertools import permutations
from scipy import optimize

def compute_acc(cs_true, cs_est, C):
    """compute the clustering accuracy"""

    # compute the linear assignment
    cs_true,cs_est = cs_true.flatten(),cs_est.flatten()
    cm = np.zeros((C,C))
    for i in range(C):
        for j in range(C):
            est_i_idx = (cs_est == i).astype(int)
            true_j_idx = (cs_true == j).astype(int)
            cm[i,j] = -np.sum(est_i_idx == true_j_idx)
    _, matchidx = optimize.linear_sum_assignment(cm)
    cs_est_trans = np.array([matchidx[s] for s in cs_est])
    return np.mean((cs_est_trans == cs_true).astype(int)), matchidx

def compute_min_A_err(A, A_est):
    min_A_err = np.inf
    min_permutation = None
    for p in permutations(range(A.shape[0])):
        p = np.array(p)
        A_p = A[p,:][:,p]
        A_err = np.abs(A_p - A_est).mean()
        if A_err < min_A_err:
            min_A_err = A_err
            min_permutation = p
    return min_A_err, min_permutation