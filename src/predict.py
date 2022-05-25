import joblib
import numpy as np
import pandas as pd

#defining parameters for iteration
n_folds = 10
cust_model_name = "tabnet_adam"

def take_input(data):
    export_df = {}
    for col in data.keys():
        value = None
        while True or value != "q":
            print("________________________________________________________________________________________________________________________________")
            print("Available options for the column are:: ____")
            print(f"{data[col]['uval']}")
            value = input(f"Enter value for {col}::>>>")
            print("________________________________________________________________________________________________________________________________")
            if value in data[col]['uval']:
                export_df[col] = value
                break
    return export_df

def custom_predictor(fold, data, target):
    Model = joblib.load(f"../divided_trained_models/{cust_model_name}/fold_{i}/{i}_model.z")


data = joblib.load("../deposition/col_data_pred.z")


