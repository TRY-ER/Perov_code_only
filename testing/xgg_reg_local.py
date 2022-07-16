from random import random
import pandas as pd
import numpy as np
from torch import square
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

IN_PATH = "./trainable_scaled_reg.csv"
TAR_COL = "JV_default_PCE_numeric"
N_SPLITS = 10
THRESH  = 9

''' Testing the script on the local machine'''

# loading the dataset 
main_df = pd.read_csv(IN_PATH)
Y = main_df[TAR_COL]
X = main_df.drop([TAR_COL],axis=1)

regressor = XGBRegressor(n_estimators=600,
                        learning_rate = 0.01,
                        booster = "gbtree",
                        verbosity = 2,
                        tree_method="gpu_hist",
                        predictor = "cpu_predictor")

skf = KFold(n_splits=N_SPLITS,shuffle=True,random_state=123)
acc_list = []
for i,(train_index, test_index) in enumerate(skf.split(X,Y)):
    if i <= THRESH: 
        print(f"[{i}/{N_SPLITS-1}] training started.....")
        X_train, Y_train = X.iloc[train_index,:].to_numpy(dtype=np.float64), Y.iloc[train_index].to_numpy(dtype=np.float64)
        X_test, Y_test = X.iloc[test_index,:].to_numpy(dtype=np.float64), Y.iloc[test_index].to_numpy(dtype=np.float64)
        regressor.fit(X_train, Y_train,
                    eval_set = [(X_test,Y_test)],
                    eval_metric = ["rmse"])
        pred = regressor.predict(X_test)
        error = mean_squared_error(Y_test, pred, squared=False)
        print(f"mse_loss : {error}")
        acc_list.append(error)
    else: break

print(acc_list)


