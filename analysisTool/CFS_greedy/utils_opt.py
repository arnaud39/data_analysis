import numpy as np
from numba import njit
from numba.typed import List as to_numba_list
from numba.core import types
from numba.typed import Dict


@njit(parallel=False)
def elog(x: float):
    if x <= 0 or x >= 1:
        return 0
    else:
        return x * np.log(x)


@njit(parallel=False)
def hist(sx: np.array):
    """
    Normalized histogram from list of samples.
    """

    d_ = Dict.empty(
        key_type=types.float64,
        value_type=types.int64,
    )
    for s in sx:
        d_[s] = d_.get(s, 0) + 1
    return (1 / len(sx)) * np.array(list(d_.values()))


@njit(parallel=False)
def hist_tuple(sx):
    """
    Normalized histogram from list of samples.
    """

    d_ = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for s in sx:
        h_s = hash(s)
        d_[h_s] = d_.get(h_s, 0) + 1
    return (1 / len(sx)) * np.array(list(d_.values()))


@njit(parallel=False)
def entropy_d(sx: np.array, base=2):
    """
    Discrete entropy estimator from list of samples.
    """
    return -np.sum(np.array(list(map(elog, hist(sx))))) / np.log(base)


@njit(parallel=False)
def entropy_d_tuple(sx, base=2):
    """
    Discrete entropy estimator from list of samples.
    """
    return -np.sum(np.array(list(map(elog, hist_tuple(sx))))) / np.log(base)


@njit(parallel=False)
def information_gain(
    f1: np.array, f2: np.array, ef1: float, ef2: float
) -> float:
    """
    Compute the information gain, where ig(f1,f2) = H(f1) - H(f1|f2)
    """
    ig = -entropy_d_tuple(to_numba_list(zip(f1, f2))) + ef1 + ef2
    return ig


@njit(parallel=False)
def su_calculation(f1: np.array, f2: np.array) -> float:
    """
    Compute the symmetrical uncertainty, where
    su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    """
    # calculate entropy of f1, t2 = H(f1)
    ef1 = entropy_d(f1)
    # calculate entropy of f2, t3 = H(f2)
    ef2 = entropy_d(f2)
    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    t1 = information_gain(f1, f2, ef1, ef2)
    # su(f1,f2) = 2*t1/(t2+t3)
    su = 2.0 * t1 / (ef1 + ef2)

    return su
