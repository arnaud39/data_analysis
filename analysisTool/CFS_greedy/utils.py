from typing import List
from scipy.stats import entropy
from numba import njit
import numpy as np
import pandas as pd

# Discrete estimators
@njit(parallel=False)
def entropy_d(sx: pd.Series):
    """
    Discrete entropy estimator from array-like of samples.
    """
    # _, cardinality = np.unique(sx, return_counts=True)
    # normalized_cardinality = cardinality/len(sx)

    normalized_cardinality = sx.value_counts(normalize=True).to_numpy()
    return np.sum((-1) * normalized_cardinality * np.log2(normalized_cardinality))


@njit(parallel=False)
def su_calculation(f1: pd.Series, f2: pd.Series) -> float:
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
    return entropy_d(pd.concat([f1, f2], axis=1))
