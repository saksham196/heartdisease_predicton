def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def load_scaler(scaler_path):
    import joblib
    return joblib.load(scaler_path)

def load_expected_columns(columns_path):
    import joblib
    return joblib.load(columns_path)

def preprocess_input(raw_input, expected_columns):
    import pandas as pd
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[expected_columns]