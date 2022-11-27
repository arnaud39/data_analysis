import pandas as pd
import time

from analysisTool import cfs, su_calculation


def pipeline():
    df_x = pd.read_csv("df_x.csv", index_col=0)
    df_y = pd.read_csv("df_y.csv", index_col=0)

    print(df_x)
    print(df_y)
    
    st = time.time()
    cfs(df_x, df_y, parallel=False)
    et = time.time()
    print(f"not parallel = {et - st} seconds")
    
    st = time.time()
    cfs(df_x, df_y, parallel=True)
    et = time.time()
    print(f"parallel = {et - st} seconds")

    """return {
        output: list(zip(*cfs(df_x, df_y[output], min_features=len(df_x.columns) - 1)))
        for output in df_y.columns
    }"""


if __name__ == "__main__":
    pipeline()
