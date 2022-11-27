from typing import List
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool
from .utils import su_calculation
from numba import njit

import numpy as np
import pandas as pd


def merit_calculation(X: pd.DataFrame, y: pd.DataFrame) -> float:
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    """

    n_samples, n_features = X.shape

    rff, rcf = 0, 0
    for i in range(n_features):
        f_i = X.iloc[:, i]

        next_rcf = 0

        # take the average
        for label in y.columns:
            y_j = y.loc[:, label]
            next_rcf += su_calculation(f_i, y_j)
        rcf /= len(y.columns)
        rcf += next_rcf

        for j in range(n_features):
            if j > i:
                f_j = X.iloc[:, j]
                rff += su_calculation(f_i, f_j)
    merits = rcf / np.sqrt(n_features + rff)
    return merits



def isUpping(A: List[float]):
    """
    This function check if the serie is increasing.
    """
    return all(A[i] <= A[i + 1] for i in range(len(A) - 1))


def cfs(
    X_: pd.DataFrame, y_: pd.DataFrame, min_features: int = 5, parallel: bool = True
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
    # F store the index of features
    features = []
    # M stores the merit values
    merits = []
    availables_features = list(X_.columns)
    # progress bar
    #pbar = tqdm(total=min_features, unit="features")
    while availables_features:
        if parallel:

            #pool = Pool()
            #merit_candidates = [
            #    pool.apply(merit_calculation, args=(X_.loc[:, features + [next_]], y_))
            #    for next_ in availables_features
            #]
            pass

        elif not parallel:
            merit_candidates = [
                merit_calculation(X_.loc[:, features + [next_]], y_)
                for next_ in availables_features
            ]
        next_merit = max(merit_candidates)
        next_feature = availables_features[merit_candidates.index(next_merit)]

        features.append(next_feature)
        merits.append(next_merit)

        availables_features.remove(next_feature)

        #pbar.update(1)
        #pbar.set_description("Added {}".format(next_feature))
        # converge criterion with greedy
        if len(features) >= min_features and not (isUpping(merits[min_features - 1 :])):
            best = merits.index(max(merits[min_features:])) + 1
            features = features[:best]
            break

    #pbar.close()

    return features, merits
