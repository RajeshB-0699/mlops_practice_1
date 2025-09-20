import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

#model = pickle.load(open('model.pkl','rb'))
def load_model(filepath: str) :
    try:
        with open(filepath,'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Error in loading model from {filepath} : {e}")

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from {filepath} : {e}")

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = df.drop(columns = ['Potability'], axis=1)
        y_test = df['Potability']
        return X_test, y_test
    except Exception as e:
        raise Exception(f"Error in preparing data : {e}")

def evaluation_model(model, X: pd.DataFrame, y:pd.Series)-> dict:
    try:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        pre = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics_dict = {
            'acc':acc,
            'pre':pre,
            'recall': recall,
            'f1':f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error in evaluating model : {e}")

def save_metrics(metrics_dict : dict, filepath : str)->None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics_dict, file, indent = 4)
    except Exception as e:
        raise Exception(f"Error in saving metrics : {e}")


# test_processed = pd.read_csv('./data/processed/test_processed.csv')

# X_test = test_processed.drop(columns = ['Potability'], axis =1)
# y_test = test_processed['Potability']

# y_pred = model.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# pre = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# metrics_dict = {
#     'acc':acc,
#     'pre':pre,
#     'recall': recall,
#     'f1':f1
# }

# with open('metrics.json','w') as file:
#     json.dump(metrics_dict,file, indent=4)

def main():
    try:
        test_processed_filepath = "./data/processed/test_processed.csv"
        model_name = 'model.pkl'
        metrics_filepath = 'metrics.json'

        test_processed = load_data(test_processed_filepath)
        X_test, y_test = prepare_data(test_processed)
        model = load_model(model_name)
        metrics_dict = evaluation_model(model, X_test , y_test)
        save_metrics(metrics_dict, metrics_filepath)

    except Exception as e:
        raise Exception(f"Error in model evaluation : {e}")
    
if __name__ == "__main__":
    main()

