import joblib
import numpy as np
import pandas as pd


Model = joblib.load("../divided_trained_models/tabnet_adam/fold_0/0_model.z")


data = joblib.load("../deposition/col_data_pred.z")

new_df= {}

for col, value in data.items():
    new_df[col] = [value["min"]]

df = pd.DataFrame(new_df,columns=new_df.keys())
tar_col = "PCE_categorical"
use_df = df.drop([tar_col],axis=1)

for col in use_df.columns:
    try:
        lbl = joblib.load(f"../outputs/smote_label_enc/{col}.z")
        use_df[col] = lbl.transform(use_df[col])
    except:pass

# scaling the dataframe
scaler = joblib.load("../outputs/scaler/standard_scaler.z")

use_df = scaler.transform(use_df)

print(use_df[0])

pred = Model.predict(use_df)
pred = np.array(pred,dtype=np.int64)
print(pred)

tar_lbl = joblib.load(f"../outputs/smote_label_enc/{tar_col}.z")

value = tar_lbl.inverse_transform(pred)

print(value)