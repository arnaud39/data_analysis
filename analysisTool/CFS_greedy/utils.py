from typing import List, Tuple
from scipy.stats import entropy
from numba import njit
from numba.experimental import jitclass
from numba.typed import Dict
from numba.typed import List as to_numba_list
from numba.core import types
import numpy as np
import pandas as pd

cache = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)


@jitclass([("cache", types.DictType(types.unicode_type, types.float64))])
class Entropy:
    def __init__(self):
        self.cache = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,
        )

    def hist(self, sx: np.array):
        """
        Normalized histogram from list of samples.
        """

        d_ = Dict.empty(
            key_type=types.float32,
            value_type=types.int64,
        )
        for s in sx:
            d_[s] = d_.get(s, 0) + 1
        return (1 / len(sx)) * np.array(list(d_.values()))

    def hist_tuple(self, sx):
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

    # Discrete estimators
    def entropy_d(self, sx: np.array, sx_id: str):
        """
        Discrete entropy estimator from array-like of samples.
        """

        if sx_id not in self.cache:
            normalized_cardinality = self.hist(sx)
            self.cache[sx_id] = np.sum(
                (-1) * normalized_cardinality * np.log2(normalized_cardinality)
            )
        return self.cache[sx_id]

    def entropy_d_tuple(self, sx):
        """
        Discrete entropy estimator from list of samples (tuple like).
        """

        normalized_cardinality = self.hist_tuple(sx)
        return np.sum((-1) * normalized_cardinality * np.log2(normalized_cardinality))

    def joint_entropy(self, f1: np.array, f2: np.array):
        """Compute the joint entropy."""

        return self.entropy_d_tuple(to_numba_list(zip(f1, f2)))


@njit(parallel=False)
def su_calculation(f1: np.array, f2: np.array, variables: Tuple[str, str], entropy_estimator) -> float:
    """
    Compute the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    """
    entropy_instance = Entropy()
    f1_id, f2_id = variables
    entropy_f1 = entropy_instance.entropy_d(f1, f1_id)

    entropy_f2 = entropy_instance.entropy_d(f2, f2_id)

    entropy_f1f2 = entropy_instance.joint_entropy(f1, f2)

    # Compute the mutual information, where ig(f1,f2) = H(f1) + H(f2) - H(f1;f2)
    mutual_information = entropy_f1 + entropy_f2 - entropy_f1f2

    su = 2.0 * mutual_information / (entropy_f1 + entropy_f2)

    return su
