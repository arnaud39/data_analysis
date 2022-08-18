import pandas as pd

from analysisTool import cfs

def pipeline():
    df_x = pd.read_csv("df_x.csv")
    df_y = pd.read_csv("df_y.csv")
    return {
        output: list(zip(*cfs(df_x, df_y[output], min_features=len(df_x.columns) - 1)))
        for output in df_y.columns
    }

if __name__ == "__main__":
    pipeline()