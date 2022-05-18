import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import optuna as opt
import os
import joblib
from datetime import datetime


# space for hyper parameters declaration

def make_save_cv_model(i,model_name,model,best_params,optim,output_path="../deposition/cross_validated_models"):

    ''' This function saves cross validation model in the corresponding directory ( if the path does not exist it creates the path for it'''


    if os.path.exists(os.path.join(f"{output_path}/{i}_{model_name}_{optim}")):
        joblib.dump(model, os.path.join(f"{output_path}/{i}_{model_name}_{optim}/{i}_model.z"))
        with open(os.path.join(f"{output_path}/{i}_{model_name}_{optim}/model_params.txt"),"w+") as file:
            file.write(best_params)
    else:
        os.mkdir(os.path.join(f"{output_path}/{i}_{model_name}_{optim}"))
        joblib.dump(model, os.path.join(f"{output_path}/{i}_{model_name}_{optim}/{i}_model.z"))
        with open(os.path.join(f"{output_path}/{i}_{model_name}_{optim}/model_params.txt"),"w+") as file:
            file.write(best_params)

def train(model_name,sc_df,tar_col,optim,k_folds=10,tar_cols="",verbose=1):

    ''' this function is used to train the model with parameters optimization using optuna and cross validation using stratified k_folds'''

    y = sc_df[tar_col]
    x = sc_df.drop([tar_col],axis=1)
    # k_fold constructing the cross-validation framework
    skf = StratifiedKFold(n_splits=k_folds,shuffle=True, random_state=123 )
    model_name = model_name 
    for i, (train_index, test_index) in enumerate(skf.split(x,y)):  
        print(f" Start time of fold {i} : {datetime.now()}") 
        def objective(trial):
            clf = SVC(C=trial.suggest_categorical("C",[1,2,3]),
                                kernel=trial.suggest_categorical("svc_kernel",["linear", "poly", "rbf", "sigmoid"]),
                                gamma = "auto",
                                decision_function_shape=trial.suggest_categorical("svc_decision_function_shape",["ovo","ovr"]),
                                verbose = 1,
                                random_state = 123,
                                )
            # print(f" train_index :: {train_index}")
            # print(f" test_index :: {test_index}")
            X_train,X_test = x.iloc[train_index,:], x.iloc[test_index,:]
            # print(X_train.shape, X_test.shape)
            X_train, X_test = X_train.to_numpy(dtype=np.float64), X_test.to_numpy(dtype=np.float64)
            Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
            Y_train, Y_test = Y_train.to_numpy(dtype=np.float64), Y_test.to_numpy(dtype=np.float64)
            print(Y_train.shape, Y_test.shape)
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            print(classification_report(Y_test, Y_pred, labels=[x for x in range(6)]))
            clf_report = classification_report(Y_test, Y_pred, labels=[x for x in range(6)])
            joblib.dump(clf_report,f"../deposition/classification_report/comp/{i}_{model_name}_classification_report.z")
            with open(f"../deposition/classification_report/{model_name}_{i}_classification_report.txt","w+") as file:file.write(clf_report)
            print(f"Saved classification_report at : deposition/classification_report/{model_name}_{i}_classification_report.txt")
            acc = accuracy_score(Y_pred, Y_test)
            return acc

        print(f"Starting optimization for fold : [{i}/{k_folds}]")
        study = opt.create_study(direction='maximize')
        study.optimize(objective, n_trials=15)
        best_params = study.best_params
        print(f" Best params for fold : [{i}/{k_folds}]")
        print(best_params)
        joblib.dump(best_params,f"../deposition/best_params/comp/{model_name}_fold_{i}_best_params.z")
        with open(f"../deposition/best_params/{model_name}_fold_{i}_best_params.txt", "w+") as file:file.write(str(best_params))
        clf_model = SVC(best_params)
        try:
            print("[++] Saving the model and parameters in corresponding directories")
            make_save_cv_model(i,model_name,clf_model,best_params,optim=optim)
        except:
            print("[-] Failed to save the model")
        print(f" End time of fold {i} : {datetime.now()}")





if __name__ == '__main__':
    use_df = pd.read_csv("../outputs/data/trainable_scaled_balanced.csv")
    tar_col = "PCE_categorical"
    model_name = "svc_classifier"
    optimizer = "un_optim"
    folds = 10
    train(model_name=model_name,
        sc_df=use_df,
        tar_col=tar_col,
        optim=optimizer,
        k_folds=folds)
    print("[++] Ended the training process ...")