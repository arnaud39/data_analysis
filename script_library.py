import pandas as pd

from analysisTool import pipeline


def pipeline():
    df_x = pd.read_csv("df_x.csv", index_col=0)
    df_y = pd.read_csv("df_y.csv", index_col=0)

    result, merits = pipeline(df_x, df_y)
    print(result)


if __name__ == "__main__":
    pipeline()
