from typing import List
from scipy.stats import entropy
from numba import njit
from numba.typed import Dict
from numba.core import types
import numpy as np
import pandas as pd



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


# Discrete estimators
@njit(parallel=False)
def entropy_d(sx: np.array):
    """
    Discrete entropy estimator from array-like of samples.
    """
    normalized_cardinality = hist(sx)
    
    #deprecated: return coutns is not supported in numba
    
    # _, cardinality = np.unique(sx, return_counts=True)
    #normalized_cardinality = cardinality / len(sx)

    #pd series version, deprecated
    # normalized_cardinality = sx.value_counts(normalize=True).to_numpy()
    return np.sum((-1) * normalized_cardinality * np.log2(normalized_cardinality))


@njit(parallel=False)
def su_calculation(f1: np.array, f2: np.array) -> float:
    """
    Compute the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    """
    entropy_f1 = entropy_d(f1)
    entropy_f2 = entropy_d(f2)
    entropy_f1f2 = joint_entropy(f1, f2)

    # Compute the mutual information, where ig(f1,f2) = H(f1) + H(f2) - H(f1;f2)
    mutual_information = entropy_f1 + entropy_f2 - entropy_f1f2

    su = 2.0 * mutual_information / (entropy_f1 + entropy_f2)

    return su


@njit(parallel=False)
def joint_entropy(f1: np.array, f2: np.array):
    """Compute the joint entropy."""
    return entropy_d(np.c_[f1, f2])
