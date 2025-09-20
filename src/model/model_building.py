import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

#est = yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']
def load_estimators(filepath: str) -> int:
    try:
        with open(filepath,'r') as file:
            est = yaml.safe_load(file)
            n_estimators =  est['model_building']['n_estimators']
            return n_estimators
    except Exception as e:
        raise Exception(f"Error in loading estimators {filepath} : {e}")

def load_data(filepath: str):
    return pd.read_csv(filepath)

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns = ['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error in preparing data : {e}")

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X,y)
        return clf
    except Exception as e:
        raise Exception(f"Error in training model : {e}")

def save_model(model: RandomForestClassifier, filepath:str) -> None:
    try:
        with open(filepath,"wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error in saving model : {e}")

def main():
    try:
        params_filepath = "params.yaml"
        processed_data_filepath = "./data/processed/train_processed.csv"
        model_save_path = "model.pkl"

        n_estimators = load_estimators(params_filepath)
        
        train_processed_data = load_data(processed_data_filepath)  
        X_train, y_train = prepare_data(train_processed_data)
        clf = train_model(X_train, y_train, n_estimators)
        save_model(clf, model_save_path)
    except Exception as e:
        raise Exception(f"Error in model evaluation : {e}")

# train_processed = pd.read_csv('./data/processed/train_processed.csv')
# test_processed = pd.read_csv('./data/processed/test_processed.csv')

# X_train = train_processed.drop(columns = ['Potability'],axis=1)
# y_train = train_processed['Potability']

# clf = RandomForestClassifier(n_estimators = est)
# clf.fit(X_train, y_train)

# pickle.dump(clf, open('model.pkl','wb'))

if __name__ == "__main__":
    main()
