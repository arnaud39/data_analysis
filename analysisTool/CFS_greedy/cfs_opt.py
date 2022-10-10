from .utils_opt import su_calculation

import numba
import numpy as np

from numba import njit, prange


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
    rff, rcf = 0, 0
    for i in prange(n_features):
        f_i = X[:, i]
        rcf += su_calculation(f_i, y)
        for j in prange(n_features):
            if j > i:
                f_j = X[:, j]
                rff += su_calculation(f_i, f_j)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


@njit(parallel=False)
def isUpping(A: np.array):
    """
    This function check if the serie is increasing.
    """
    for i in range(len(A) - 1):
        if A[i] > A[i + 1]:
            return False
    return True


@njit(signature, parallel=False)
def cfs_opt(X: np.array, y: np.array, min_features):
    """
    This function uses a correlation based greedy to evaluate the
    worth of features.
    The algorithm works as following:
    - at each iteration we will add the best feature to the candidate set
    regarding the heuristic function defined in Chapter 4
    Correlation-based Feature Selection of given refenrence.
    - we stop of the algorithm is based on convergence
    - one can specify the minimum number of features
    Mark A. Hall, "Correlation-based Feature Selection
    for Machine Learning", 1999.
    """

    n_features = len(X[0])
    # F store the index of features
    features = [0]
    features.pop()
    # M stores the merit values
    merits = [1.0]
    merits.pop()
    best = 0  # initialize best feature until no convergence
    availables_features = list(range(n_features))

    while availables_features:
        merit_candidates = []
        for next_ in availables_features:
            features.append(next_)
            merit_candidates.append(merit_calculation(X[:, np.array(features)],
                                                      y))
            features.pop()
        next_merit = max(merit_candidates)
        next_feature = availables_features[merit_candidates.index(next_merit)]

        features.append(next_feature)
        merits.append(next_merit)

        availables_features.remove(next_feature)

        # converge criterion with greedy
        if (len(features) >= min_features) and not (
            isUpping(merits[min_features-1:])
        ):
            best = merits.index(max(merits[min_features:])) + 1
            features = features[:best]
            break

    features_array = np.array(features)
    merits_array = np.array(merits)
    return features_array, merits_array
