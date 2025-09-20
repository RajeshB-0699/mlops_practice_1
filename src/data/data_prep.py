import pandas as pd
import numpy as np
import os

# train_data = pd.read_csv('./data/raw/train.csv')
# test_data = pd.read_csv('./data/raw/test.csv')

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading file from {filepath} : {e}")

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
        return df

# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

# processed_path = os.path.join("data","processed")
# os.makedirs(processed_path)

# train_processed_data.to_csv(os.path.join(processed_path,"train_processed.csv"), index=False)
# test_processed_data.to_csv(os.path.join(processed_path, "test_processed.csv"), index=False)

def save_file(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error in saving file {filepath} : {e}")

def main():
    try:
        train_raw_datapath = "./data/raw/train.csv"
        test_raw_datapath = "./data/raw/test.csv"

        train_raw_data = load_data(train_raw_datapath)
        test_raw_data = load_data(test_raw_datapath)

        train_processed_data = fill_missing_with_median(train_raw_data)
        test_processed_data = fill_missing_with_median(test_raw_data)

        processed_path = os.path.join("data", "processed")
        os.makedirs(processed_path)

        save_file(train_processed_data, os.path.join(processed_path, "train_processed.csv"))
        save_file(test_processed_data, os.path.join(processed_path, "test_processed.csv"))
    except Exception as e:
        raise Exception("Error in data prep : {e}")
if __name__ == "__main__":
    main()





