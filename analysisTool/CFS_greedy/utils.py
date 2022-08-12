from typing import List
from math import log
from scipy.stats import entropy

import numpy as np

# Discrete estimators
def entropy_d(sx: List[float], base=2):
    """
    Discrete entropy estimator from list of samples.
    """
    d_ = dict()
    d_ = {s: d_.get(s,0)+1 for s in sx}
    return -sum(map(elog, map(lambda z: float(z)/len(sx), d_.values())))/log(base)

def su_calculation(f1: np.array, f2: np.array) -> float:
    """
    Compute the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    """
    a = -entropy_d(list(zip(f1,f2)))
    b = entropy_d(f1)
    c = entropy_d(f2)
    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    # calculate entropy of f1, t2 = H(f1)
    # calculate entropy of f2, t3 = H(f2)
    # su(f1,f2) = 2*t1/(t2+t3)
    return 2.0*(a+b+c)/(b+c)


def elog(x: float):
    if x <= 0 or x >= 1:
        return 0
    return x*log(x)
