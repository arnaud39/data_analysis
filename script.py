import pandas as pd
import time
import numpy as np


from analysisTool import cfs
from typing import List, Tuple


def find_segment(number, array_) -> int:
    """Find associated segment of a number in a given sorted array."""
    if number <= array_[0]:
        return 0
    elif number >= array_[-1]:
        return len(array_) - 2
    for segment_id, (inf_, sup_) in enumerate(zip(array_[:-1], array_[1:])):
        if number >= inf_ and number < sup_:
            return segment_id


def convert_objects(df: pd.DataFrame) -> Tuple[List[str], np.array]:
    """
    Inplace conversion of dataframes to hashed values.
    Here, the hash function is bijectiv.
    Return the list of hashed columns.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    objects_cols = df.select_dtypes(exclude=numerics).columns
    for proc_col in objects_cols:
        bijectiv_hasher = {k: v for v, k in enumerate(df[proc_col].unique())}
        df[proc_col] = df[proc_col].replace(bijectiv_hasher).astype("int32")
    return list(objects_cols), df[objects_cols].to_numpy()


def convert_numbers(df: pd.DataFrame) -> Tuple[List[str], np.array]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numeric_cols = df.select_dtypes(include=numerics).columns

    stack = []
    for proc_col in list(numeric_cols):
        mean_, std_ = df[proc_col].mean(), df[proc_col].std()

        segments = np.linspace(mean_ - 5 * std_, mean_ + 5 * std_, 255)  # int 8bits
        stack.append(np.digitize(df[proc_col].to_numpy(), segments).astype("int32"))

    return list(numeric_cols), np.column_stack(tuple(stack))


def pipeline():
    st = time.time()
    df_x = pd.read_csv("df_x.csv", index_col=0)
    df_y = pd.read_csv("df_y.csv", index_col=0)

    # remove nan, that would be used for entropy calculation
    index_drop = df_x.isna().any(axis=1)
    df_x = df_x[~index_drop]
    df_y = df_y[~index_drop]

    ojects_x, x_objects = convert_objects(df_x)
    objects_y, y_objects = convert_objects(df_y)

    numbers_x, x_numbers = convert_numbers(df_x)
    numbers_y, y_numbers = convert_numbers(df_y)
    
    x_cols = ojects_x + numbers_x
    
    
    X = np.hstack((x_objects, x_numbers)).astype("uint8")
    y = np.hstack((y_objects, y_numbers)).astype("uint8")
    et = time.time()
    print(f"pipeline: {et - st} seconds")

    st = time.time()
    features_array, merits_array = cfs(X, y, 5)
    et = time.time()
    print(f"cfs: {et - st} seconds")
    
    #features = [x_cols[k] for k in features_array]

    #print(features)
    print(merits_array)


if __name__ == "__main__":
    pipeline()
