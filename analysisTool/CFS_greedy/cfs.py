from typing import List
from tqdm import tqdm
from .utils import su_calculation

import numpy as np
import pandas as pd

def merit_calculation(X: np.array, y: np.array) -> float:
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    """

    n_samples, n_features = X.shape
    rff, rcf = 0, sum([su_calculation(X[:,i],y) for i in range(n_features)])
    for i in range(n_features):
        for j in range(i+1,n_features):
            rff += su_calculation(X[:,i], X[:,j])
    rff *= 2
    return rcf / np.sqrt(n_features + rff)


def isUpping(A: List[float]):
    """
    This function check if the serie is increasing.
    """
    return all(A[i] <= A[i + 1] for i in range(len(A) - 1))


def cfs(X_: pd.DataFrame, y_: pd.DataFrame, min_features: int=5) -> np.array:
    """
    This function uses a correlation based greedy to evaluate the worth of features.
    
    The algorithm works as following: 
    - at each iteration we will add the best feature to the candidate set regarding the heuristic function defined in 
    Chapter 4 Correlation-based Feature Selection of given refenrence.
    - we stop of the algorithm is based on convergence
    - one can specify the minimum number of features
    
    Mark A. Hall "Correlation-based Feature Selection for Machine Learning" 1999.
    """

    X, y = X_.to_numpy(), y_.to_numpy().squeeze()
    n_samples, n_features = X.shape
    # F store the index of features
    # M stores the merit values
    features, merits = [], []
    availables_features = [k for k in range(n_features)]
    #progress bar
    pbar = tqdm(total=min_features, unit='features')
    while availables_features:
        merit_candidates = [
            merit_calculation(X[:, features + [next_]], y)
            for next_ in availables_features
        ]
        next_merit = max(merit_candidates)
        next_feature = availables_features[merit_candidates.index(next_merit)]

        features.append(next_feature)
        merits.append(next_merit)

        availables_features.remove(next_feature)
        
        pbar.update(1)
        pbar.set_description("Added {}".format(X_.columns[next_feature]))
        # converge criterion with greedy
        if len(features) >= min_features and not (isUpping(merits[min_features-1:])):
            best = merits.index(max(merits[min_features:])) + 1
            features = features[:best]
            break
            
    pbar.close()
            
    return list(map(lambda id_: X_.columns[id_], features)), merits