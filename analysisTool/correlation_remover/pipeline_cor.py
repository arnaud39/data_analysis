from .corr_ import remove_features
from typing import List

import pandas as pd
import numpy as np


def pipeline_cor(df_x: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """Remove redundant features from a given dataframe.
    Note: we perform a linear correlation based analysis here.
    This function is intended to be a pre-filter for the cfs algorithm.

    Args:
        df_x (pd.DataFrame): Dataframe of features

    Returns:
        List[str]: List of features to be kept.
    """
    corr_matrix = df_x.corr().abs()
    df_corr = (
        (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) > 0.9)
        .stack()
        .sort_values(ascending=False)
        .reset_index()
    )
    dict_ = df_corr[df_corr[df_corr.columns[-1]] == True][
        ["level_0", "level_1"]
    ].to_dict("index")
    set_ = set(map(lambda x: frozenset(x.values()), dict_.values()))

    initial_features = set(df_x.columns)
    features_to_remove = remove_features(set_)

    new_features = initial_features.difference(*list(features_to_remove))
    return new_features
