from typing import List
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool
from .utils import su_calculation
from numba import njit, prange

import numba
import numpy as np
import pandas as pd

signature = (numba.typeof((np.array([np.int64(1)]), np.array([0.1, 0.1]))))(
    numba.types.Array(dtype=numba.types.float64, ndim=2, layout="C"),
    numba.types.Array(dtype=numba.types.float64, ndim=1, layout="C"),
    numba.typeof(5),
)


@njit(parallel=True)
def merit_calculation(X: np.array, y: np.array) -> float:
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    """

    n_samples, n_features = X.shape
    #_, n_labels = y.shape

    rff, rcf = 0, 0
    for i in prange(n_features):
        f_i = X[:, i]

        next_rcf = 0

        # take the average
        for label_index in range(1):
            y_j = y#[:, label_index]
            next_rcf += su_calculation(f_i, y_j)
        next_rcf /= 1#n_labels
        rcf += next_rcf

        for j in range(n_features):
            if j > i:
                f_j = X[:, j]
                rff += su_calculation(f_i, f_j)
    merits = rcf / np.sqrt(n_features + rff)
    return merits


@njit(parallel=False)
def isUpping(A: np.array):
    """
    This function check if the serie is increasing.
    """
    return all(A[i] <= A[i + 1] for i in range(len(A) - 1))


@njit(signature, parallel=False)
def cfs(
    X_: np.array,
    y_: np.array,
    min_features: int = 5,
) -> np.array:
    """
    This function uses a correlation based greedy to evaluate the worth of features.

    The algorithm works as following:
    - at each iteration we will add the best feature to the candidate set regarding the heuristic function defined in
    Chapter 4 Correlation-based Feature Selection of given refenrence.
    - we stop of the algorithm is based on convergence
    - one can specify the minimum number of features

    Mark A. Hall "Correlation-based Feature Selection for Machine Learning" 1999.
    """

    # X, y = X_.to_numpy(), y_.to_numpy().squeeze()
    n_samples, n_features = X_.shape
    # index of features
    features = []
    # merit values
    merits = []
    availables_features = list(range(n_features))

    while availables_features:
        merit_candidates = []
        for next_ in availables_features:
            features.append(next_)
            merit_candidates.append(merit_calculation(X_[:, np.array(features)], y_))
            features.pop()
        next_merit = max(merit_candidates)
        next_feature = availables_features[merit_candidates.index(next_merit)]

        features.append(next_feature)
        merits.append(next_merit)

        availables_features.remove(next_feature)

        # converge criterion with greedy
        if len(features) >= min_features and not (isUpping(merits[min_features - 1 :])):
            #best = merits.index(max(merits[min_features:])) + 1
            #features = features[:best]
            break

    features_array = np.array(features)
    merits_array = np.array(merits)
    return features_array, merits_array
