from .CFS_greedy.pipeline_cfs import pipeline_cfs
from .correlation_remover.pipeline_cor import pipeline_cor

from typing import Tuple, List

import pandas as pd
import numpy as np

def pipeline(df_x: pd.DataFrame, df_y: pd.DataFrame) -> Tuple[List[str], np.array]:
    """_summary_

    Args:
        df_x (pd.DataFrame): _description_
        df_y (pd.DataFrame): _description_

    Returns:
        Tuple[List[str], np.array]: _description_
    """

    new_features_ = pipeline_cor(df_x)
    df_x = df_x.loc[:, list(new_features_)]
    result, merits = pipeline_cfs(df_x, df_y)
    return result, merits