import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def save_k_fold(df,k_folds, tarcol,save_path="./outputs/fold_vals/"):
    ...