import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
import optuna as opt
import torch
import os
import joblib


def make_save_cv_model(i,model_name,model,best_params,optim,clf_report,accuracy,trial_data,output_path="../outputs/test_models"):

    ''' This function saves cross validation model in the corresponding directory ( if the path does not exist it creates the path for it'''


    if os.path.exists(os.path.join(output_path,f"{i}_{model_name}_{optim}")):
        joblib.dump(model, os.path.join(output_path,f"{i}_{model_name}_{optim}/{i}_model.z"))
        with open(os.path.join(output_path,f"{i}_{model_name}_{optim}/model_params.txt"),"w+") as file:file.write(str(best_params))
        with open(os.path.join(output_path,f"{i}_{model_name}_{optim}/classification_report.txt"),"w+") as file:file.write(str(clf_report))
        with open(os.path.join(output_path,f"{i}_{model_name}_{optim}/accuracy_score.txt"),"w+") as file:file.write(f" accuracy :: {str(accuracy)}")
        joblib.dump(trial_data, os.path.join(output_path,f"{i}_{model_name}_{optim}/{i}_trial_data.z"))
    else:
        os.mkdir(os.path.join(output_path,f"{i}_{model_name}_{optim}"))
        joblib.dump(model, os.path.join(output_path,f"{i}_{model_name}_{optim}/{i}_model.z"))
        with open(os.path.join(output_path,f"{i}_{model_name}_{optim}/model_params.txt"),"w+") as file:file.write(str(best_params))
        with open(os.path.join(output_path,f"{i}_{model_name}_{optim}/classification_report.txt"),"w+") as file:file.write(str(clf_report))
        with open(os.path.join(output_path,f"{i}_{model_name}_{optim}/accuracy_score.txt"),"w+") as file:file.write(f" accuracy :: {str(accuracy)}")
        joblib.dump(trial_data, os.path.join(output_path,f"{i}_{model_name}_{optim}/{i}_trial_data.z"))


def get_trial_data(trial) -> list:
  ''' This function takes the trial objects and returns the dictionary containing the trial details for plotting and comparing purposes '''
  trial_data = trial.get_trials()
  value_dict = {}
  for i in trial_data:
    print(i.params)
    value_dict[i.number] = {"params": i.params , "accuracy": i.values}
    print(f"{i.number} : {i.values}")
  return value_dict

def train(fold_dict,fold,model_name,sc_df,tar_col,optim,optim_trial,k_folds=10,tar_cols="",verbose=1):

    ''' this function is used to train the model with parameters optimization using optuna and cross validation using stratified k_folds'''

    y = sc_df[tar_col]
    print(y.shape)
    x = sc_df.drop([tar_col],axis=1)
    print(x.shape)
    model_name = model_name 
    def objective(trial):
      train_index = fold_dict[fold]["train"]
      test_index = fold_dict[fold]["test"]
      clf = XGBClassifier(n_estimators=trial.suggest_categorical("xgb_est",[100,200,300,400,500]),
                        learning_rate=trial.suggest_categorical("xgb_lr",[0.1,0.01,0.001]),
                        booster = trial.suggest_categorical("xgb_booster",["gbtree","gblinear","dart"]),
                        tree_method = "gpu_hist",
                        predictor = "gpu_predictor")
      # print(f" train_index :: {train_index}")
      # print(f" test_index :: {test_index}")
      X_train,X_test = x.iloc[train_index,:], x.iloc[test_index,:]
      # print(X_train.shape, X_test.shape)
      X_train, X_test = X_train.to_numpy(dtype=np.float64), X_test.to_numpy(dtype=np.float64)
      Y_train, Y_test = y.iloc[train_index].to_numpy(dtype=np.float64), y.iloc[test_index].to_numpy(np.float64)
      # Y_train, Y_test = Y_train.to_numpy(dtype=np.float64), Y_test.to_numpy(dtype=np.float64)
      print(X_train.shape)
      print(Y_train.shape)
      print(X_test.shape)
      print(Y_test.shape)
      clf.fit(X_train, Y_train,
              eval_set=[(X_test, Y_test)],
              eval_metric=['mlogloss'])
      Y_pred = clf.predict(X_test)
      print(classification_report(Y_test, Y_pred, labels=[x for x in range(6)]))
      acc = accuracy_score(Y_pred, Y_test)
      return acc

    print(f"Starting optimization for fold : [{fold}/{k_folds}]")
    study = opt.create_study(direction='maximize')
    study.optimize(objective, n_trials=optim_trial)
    best_params = study.best_params
    trial_data = get_trial_data(study)
    print(f" Best params for fold : [{fold}/{k_folds}]")
    print(best_params)
    train_index = fold_dict[fold]["train"]
    test_index = fold_dict[fold]["test"]
    X_train,X_test = x.iloc[train_index,:], x.iloc[test_index,:]
    # print(X_train.shape, X_test.shape)
    X_train, X_test = X_train.to_numpy(dtype=np.float64), X_test.to_numpy(dtype=np.float64)
    Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
    Y_train, Y_test = Y_train.to_numpy(dtype=np.float64), Y_test.to_numpy(dtype=np.float64)
    clf_model = XGBClassifier(**study.best_params)
    clf_model.fit(X_train,Y_train)
    Y_pred = clf_model.predict(X_test)
    clf_report = classification_report(Y_test, Y_pred, labels=[x for x in range(6)])
    accuracy = accuracy_score(Y_pred, Y_test)
    # try:
    print("[++] Saving the model and parameters in corresponding directories")
    make_save_cv_model(fold,model_name,clf_model,best_params,optim,clf_report,accuracy,trial_data=trial_data)
    return trial_data
    # except:
    #     print("[-] Failed to save the model")

use_df = pd.read_csv("../outputs/data/trainable_scaled_balanced.csv")
tar_col = "PCE_categorical"
model_name = "xg_boost"
fold_dict = joblib.load("../outputs/fold_vals_xgboost/fold_data.z")
optim = "no_optim"
fold = 0
k_folds = 20
num_trials = 2

trial_data = train(fold_dict = fold_dict,
      fold = fold,
      model_name=model_name,
      sc_df=use_df,
      tar_col=tar_col,
      optim = optim,
      optim_trial = num_trials)
for key,value in trial_data.items():
  print(f"{key}: {value['accuracy']}")
print(f"[++] Ended the training process for fold {fold}")