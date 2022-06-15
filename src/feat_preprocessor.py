import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler 

main_df = pd.read_csv("../outputs/data/feat_mod_encoded_data.csv")
tar_col = main_df["PCE_categorical"].values
main_df = main_df.drop(["PCE_categorical"],axis=1)
cols = main_df.columns 
scaler = StandardScaler()
trans_df = scaler.fit_transform(main_df)
trans_df = pd.DataFrame(trans_df,columns=cols)
trans_df["PCE_categorical"] = tar_col
trans_df.to_csv("../outputs/data/feat_scaled_data.csv",index=False)
joblib.dump(scaler, "../outputs/scaler/feat_standard_scaler.z")
print(trans_df.head())