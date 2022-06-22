import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


main_df = pd.read_csv("../inputs/smote_main_reversed_reedited.csv")
col_data = joblib.load("../inputs/columns_encoded/syn_data.z")
col_data_dict = joblib.load("../deposition/col_data_pred.z")


#we have to eliminate those duplicate values from the dataset by excluding space values either from starting or ending of the string (for categorical sampeles only) 

def elem_space(word):
    ''' This function reduces all the space occupied values to the individual values'''
    try:
        if word[0] == " ": word = word[1:]
        if word[-1] == " ": word = word[:-1:]
    except: ...
    return word

for col in main_df.columns:
    new_val = []
    for value in main_df[col].values:
        indiv_val = elem_space(value)
        new_val.append(indiv_val)
    main_df[col] = new_val

for col in col_data_dict.keys():
    if col_data_dict[col]["type"] == "bool":
       main_df[col] = main_df[col].map({True: "Yes", False: "No"})
       print(main_df[col].dtype)

main_df.to_csv("../outputs/data/space_preprocessed_main.csv",index=False)

cat_cols = col_data["categorical"]
tar_col = "PCE_categorical"


# using label encoding
trans_df = main_df.copy()
for col in main_df.columns:
    if col in cat_cols:
        print(col)
        lbl_enc = LabelEncoder()
        trans_df[col] = lbl_enc.fit_transform(trans_df[col])
        joblib.dump(lbl_enc,f"../outputs/smote_label_enc/{col}.z")
    elif col == tar_col:
        lbl_enc = LabelEncoder()
        trans_df[col] = lbl_enc.fit_transform(trans_df[col])
        joblib.dump(lbl_enc,f"../outputs/smote_label_enc/{col}.z")

trans_df.to_csv("../outputs/data/trainable_encoded.csv",index=False)

Y = trans_df[tar_col]
X = trans_df.drop([tar_col],axis=1)
scl = StandardScaler()
X_trans = scl.fit_transform(X)
joblib.dump(scl,"../outputs/scaler/standard_scaler.z")

trainable_df = pd.DataFrame(X_trans, columns=X.columns)
trainable_df[tar_col] = Y

trainable_df.to_csv("../outputs/data/trainable_scaled_balanced.csv",index=False)
