import pandas as pd


def load_essays(train_path, val_path, test_path):

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    return train_df, val_df, test_df