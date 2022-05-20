import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import joblib

def save_k_fold(df,k_folds, tarcol,save=True,save_path="./outputs/fold_vals/"):
    y = df[tarcol]
    x = df.drop([tarcol],axis=1)
    # k_fold constructing the cross-validation framework
    skf = StratifiedKFold(n_splits=k_folds,shuffle=True, random_state=123 )
    fold_dict = {}
    for i, (train_index, test_index) in enumerate(skf.split(x,y)):
        index_data = {"train": train_index,
                      "test": test_index}
        fold_dict[i] = index_data
    if save:
        joblib.dump(fold_dict,f"{save_path}fold_data.z")
        print(f"saved the fold_data at {save_path}fold_data.z")