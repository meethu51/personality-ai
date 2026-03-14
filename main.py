from src.load_data import load_essays
from src.preprocess import clean_text
from src.features import extract_features
from src.train_model import train_model

import pandas as pd


TRAIN_PATH = "data/raw/train.parquet"
VAL_PATH = "data/raw/validation.parquet"
TEST_PATH = "data/raw/test.parquet"


def main():

    # Load dataset
    train_df, val_df, test_df = load_essays(TRAIN_PATH, VAL_PATH, TEST_PATH)

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    print("Columns:", train_df.columns)

    # Clean text
    train_df["clean_text"] = train_df["text"].apply(clean_text)

    # Extract linguistic features
    print("\nExtracting linguistic features...")

    feature_rows = train_df["clean_text"].apply(extract_features)

    feature_df = pd.DataFrame(feature_rows.tolist())

    print("\nFeature preview:")
    print(feature_df.head())

    # Train models for all Big Five traits
    print("\nTraining personality models...")

    traits = ["O", "C", "E", "A", "N"]

    for trait in traits:

        print("\nTraining model for", trait)

        X = feature_df
        y = train_df[trait]

        model = train_model(X, y,trait)


if __name__ == "__main__":
    main()