import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
import optuna as opt
import torch
import os
import joblib

def make_save_cv_model(i,model_name,model,best_params,output_path="../divided_trained_models/xgboost/"):

    ''' This function saves cross validation model in the corresponding directory ( if the path does not exist it creates the path for it'''


    if os.path.exists(os.path.join(output_path,f"{i}_{model_name}")):
        joblib.dump(model, os.path.join(output_path,f"{i}_{model_name}/{i}_model.z"))
        with open(os.path.join(output_path,f"{i}_{model_name}/model_params.txt"),"w+") as file:
            file.write(str(best_params))
    else:
        os.mkdir(os.path.join(output_path,f"{i}_{model_name}"))
        joblib.dump(model, os.path.join(output_path,f"{i}_{model_name}/{i}_model.z"))
        with open(os.path.join(output_path,f"{i}_{model_name}/model_params.txt"),"w+") as file:
            file.write(str(best_params))


def train(fold_dict, fold, model_name, sc_df,tar_col, optim_trial, k_folds=20,tar_cols="", verbose=1):
    y = sc_df[tar_col]
    x = sc_df.drop([tar_col],axis=1)
    model_name = model_name
    def objective(trial):
        train_index = fold_dict[fold]["train"]
        test_index = fold_dict[fold]["test"]
        clf = XGBClassifier(n_estimators=trial.suggest_categorical("xgb_est",[100,200,300,400,500]),
                        learning_rate=trial.suggest_categorical("xgb_lr",[0.1,0.01,0.001]),
                        booster = trial.suggest_categorical("xgb_booster",["gbtree","gblinear","dart"]),
                        tree_method = "gpu_hist",
                        predictor = "cpu_predictor")
        # print(f" train_index :: {train_index}")
        # print(f" test_index :: {test_index}")
        X_train,X_test = x.iloc[train_index,:], x.iloc[test_index,:]
        # print(X_train.shape, X_test.shape)
        X_train, X_test = X_train.to_numpy(dtype=np.float64), X_test.to_numpy(dtype=np.float64)
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
        Y_train, Y_test = Y_train.to_numpy(dtype=np.float64), Y_test.to_numpy(dtype=np.float64)
        print(Y_train.shape, Y_test.shape)
        clf.fit(X_train, Y_train,
                eval_set=[(X_test, Y_test)],
                eval_metric=['auc'])
        Y_pred = clf.predict(X_test)
        print(classification_report(Y_test, Y_pred, labels=[x for x in range(6)]))
        clf_report = classification_report(Y_test, Y_pred, labels=[x for x in range(6)])
        joblib.dump(clf_report,f"../outputs/classification_report/comp/{fold}_{model_name}_classification_report.z")
        with open(f"../outputs/classification_report/{model_name}_{fold}_classification_report.txt","w+") as file:file.write(clf_report)
        print(f"Saved classification_report at : outputs/classification_report/{model_name}_{fold}_classification_report.txt")
        acc = accuracy_score(Y_pred, Y_test)
        return acc

    print(f"Starting optimization for fold : [{fold}/{k_folds}]")
    study = opt.create_study(direction='maximize')
    study.optimize(objective, n_trials=optim_trial)
    best_params = study.best_params
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
    with open(f"../divided_trained_models/xgboost/{fold}_{model_name}/{model_name}_{fold}_classification_report.txt","w+") as file:file.write(str(clf_report))
    accuracy = accuracy_score(Y_pred, Y_test)
    print(f"Accuracy :: {accuracy}")
    with open(f"../divided_trained_models/xgboost/{fold}_{model_name}/{model_name}_{fold}_accuracy_score.txt","w+") as file:file.write(f" accuracy :: {str(accuracy)}")
    try:
        print("[++] Saving the model and parameters in corresponding directories")
        make_save_cv_model(fold,model_name,clf_model,best_params)
    except:
        print("[-] Failed to save the model")



if __name__ == "__main__":
    use_df = pd.read_csv("../outputs/data/trainable_scaled_balanced.csv")
    tar_col = "PCE_categorical"
    model_name = "xg_boost"
    fold_dict = joblib.load("../outputs/fold_vals_xgboost/fold_data.z")
    fold = 0

    train(fold_dict = fold_dict,
        fold = fold,
        model_name=model_name,
        sc_df=use_df,
        tar_col=tar_col,
        optim_trial = 15)
    print(f"[++] Ended the training process for fold {fold}")