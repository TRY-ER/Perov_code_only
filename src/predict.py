import joblib
import numpy as np
import pandas as pd



def take_input(data):

    ''' This function takes an input dataset and returns another dataset with use input data'''

    export_df = {}
    for col in data.keys():
        value = None
        while True or value != "q":
            print("________________________________________________________________________________________________________________________________")
            print("Available options for the column are:: ____")
            print(f"{data[col]['u_val']}")
            value = input(f"Enter value for {col}::>>>")
            print("________________________________________________________________________________________________________________________________")
            if data[col]["type"] == "categorical":
                if value in data[col]['u_val']:  ## This comditional approach is invalid if the column is not a numeric type 
                    export_df[col] = value     ## This if loop is only designed for categorial columns
                    break
            else: 
                export_df[col] = value
                break
    return pd.DataFrame(export_df, columns=data.keys())

def custom_preprocessor(data):

    ''' this function preprocesses the data and returns the scaled version that can be used for prediction purposes
        The data input strictly needs to be in pandas dataFrame format 
        The output df will be of numpy array format'''
    
    use_df = data.copy()
    for col in data.keys():
        try:
            lbl = joblib.load(f"../outputs/smote_label_enc/{col}.z")
            use_df[col] = lbl.transform(use_df[col])
        except: ...

    scaler = joblib.load("../outputs/scaler/standard_scaler.z")
    use_df = scaler.transform(use_df)
    return use_df

def inverse_transform_target(tar_array,target):
    ''' This function transforms the integer value to corresponding label encoding value for the target
        tar_array has to be in numpy array format
        return will be an array too'''

    tar_lbl = joblib.load(f"../outputs/smote_label_enc/{target}.z")
    value = tar_lbl.inverse_transform(tar_array)
    return value

def custom_predictor(fold, data, target):

    ''' this function redicts the target with individual fold data '''
    Model = joblib.load(f"../divided_trained_models/{cust_model_name}/fold_{fold}/{fold}_model.z")
    pred = Model.predict(data)   # data must be in the pandas dataframe format
    return np.array(pred, dtype=np.int64)


if __name__ == "__main__":
    #defining parameters for iteration
    n_folds = 6
    cust_model_name = "tabnet_adam"
    data = joblib.load("../deposition/col_data_pred.z")
    target = "PCE_categorical"
    mod_df = take_input(data)
    mod_df = custom_preprocessor(mod_df)
    pred_list = []
    for i in range(n_folds): 
        prediction = custom_predictor(fold=i, data=mod_df, target=target)
        pred_list.append(prediction)
    fold_res = np.max(pred_list)
    final_res = inverse_transform_target(tar_array=fold_res, target=target)
    print(f" Final prediction is found to be {final_res}")