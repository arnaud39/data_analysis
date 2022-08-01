from typing import List
from math import log
from scipy.stats import entropy

import numpy as np

# Discrete estimators
def entropy_d(sx: List[float], base=2):
    """
    Discrete entropy estimator from list of samples.
    """
    return -sum(map(elog, hist(sx)))/log(base)

def hist(sx: List[float]):
    """
    Normalized histogram from list of samples.
    """
    
    d_ = dict()
    for s in sx:
        d_[s] = d_.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d_.values())

def su_calculation(f1: np.array, f2: np.array) -> float:
    """
    Compute the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    """
    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    t1 = information_gain(f1, f2)
    # calculate entropy of f1, t2 = H(f1)
    t2 = entropy_d(f1)
    # calculate entropy of f2, t3 = H(f2)
    t3 = entropy_d(f2)
    # su(f1,f2) = 2*t1/(t2+t3)
    su = 2.0*t1/(t2+t3)

    return su

def information_gain(f1: np.array, f2: np.array):
    """
    Compute the information gain, where ig(f1,f2) = H(f1) - H(f1|f2)
    """
    ig = entropy_d(f1) - conditional_entropy(f1, f2)
    return ig

def midd(x, y):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """

    return -entropy_d(list(zip(x, y)))+entropy_d(x)+entropy_d(y)

def conditional_entropy(f1: np.array, f2: np.array) -> float:
    """
    This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
    """

    ce = entropy_d(f1) - midd(f1, f2)
    return ce

def elog(x: float):
    if x <= 0 or x >= 1:
        return 0
    else:
        return x*log(x)