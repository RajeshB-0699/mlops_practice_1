import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

#test_size = yaml.safe_load(open('params.yaml'))['data_collection']['test_size']
def load_params(filepath : str) -> float:
    try:
        with open(filepath,'r') as file :
            params = yaml.safe_load(file)
            return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"Error in loading params from {filepath} : {e}")

#data = pd.read_csv('https://raw.githubusercontent.com/RajeshB-0699/datasets_raw/refs/heads/main/water_potability.csv')
def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from {filepath}: {e}")

#train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
def split_data(data: pd.DataFrame, test_size: float)-> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state = 42)
    except ValueError as e:
        raise Exception(f"Error in splitting data  {e}")

# datapath  = os.path.join("data","raw")
# os.makedirs(datapath)

# train_data.to_csv(os.path.join(datapath, "train.csv"),index=False)
# test_data.to_csv(os.path.join(datapath,"test.csv"),index=False)

def save_data(df:pd.DataFrame, filepath: str)-> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception("Error in saving file data : {e}")

def main():
    try:
        data_filepath = "https://raw.githubusercontent.com/RajeshB-0699/datasets_raw/refs/heads/main/water_potability.csv"
        params_filepath = "params.yaml"
        raw_data = os.path.join("data","raw")
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data,test_size=test_size)
        os.makedirs(raw_data)
        save_data(train_data, os.path.join(raw_data,"train.csv"))
        save_data(test_data, os.path.join(raw_data, 'test.csv'))
    except Exception as e:
        raise Exception(f"Error in data collection file {e}")
    

if __name__ == "__main__":
    main()




