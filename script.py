import pandas as pd
import time
import numpy as np

from analysisTool import cfs, su_calculation


def pipeline():
    df_x = pd.read_csv("df_x.csv", index_col=0)
    df_y = pd.read_csv("df_x.csv", index_col=0)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df_x = df_x.select_dtypes(include=numerics)
    index_drop = df_x.isna().any(axis=1)
    
    df_x = df_x[~index_drop]
    df_y = df_y[~index_drop]
 
    
    st = time.time()
    cfs(df_x.to_numpy(), df_y.to_numpy())
    et = time.time()
    print(f"{et - st} seconds")
    
  

    """return {
        output: list(zip(*cfs(df_x, df_y[output], min_features=len(df_x.columns) - 1)))
        for output in df_y.columns
    }"""


if __name__ == "__main__":
    pipeline()
