import joblib

fold_vals = joblib.load("../outputs/fold_vals/fold_data.z")

for key, value in fold_vals.items():
    print(f"{key} \n")
    print(f"{key} = {value['train'].shape}")
    print(f"{key} = {value['test'].shape}")