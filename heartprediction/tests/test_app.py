import streamlit as st
import pandas as pd
import joblib
import pytest

# Load the model, scaler, and expected columns for testing
model = joblib.load("models/Logistic_Regression.pkl")
scaler = joblib.load("models/scaler.pkl")
expected_columns = joblib.load("models/columns.pkl")

def test_prediction_high_risk():
    input_data = {
        'Age': 60,
        'RestingBP': 120,
        'Cholestrol': 250,
        'FastingBs': 1,
        'MaxHR': 100,
        'Oldpeak': 2.0,
        'Sex_M': 1,
        'ChestPainType_ATA': 1,
        'RestingECG_Normal': 1,
        'ExerciseAngina_Y': 1,
        'ST_slope_Up': 1
    }
    input_df = pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaler_input = scaler.transform(input_df)
    prediction = model.predict(scaler_input)[0]
    assert prediction == 1  # Expecting high risk

def test_prediction_low_risk():
    input_data = {
        'Age': 30,
        'RestingBP': 110,
        'Cholestrol': 180,
        'FastingBs': 0,
        'MaxHR': 150,
        'Oldpeak': 0.5,
        'Sex_F': 1,
        'ChestPainType_NAP': 1,
        'RestingECG_Normal': 1,
        'ExerciseAngina_N': 1,
        'ST_slope_Down': 1
    }
    input_df = pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaler_input = scaler.transform(input_df)
    prediction = model.predict(scaler_input)[0]
    assert prediction == 0  # Expecting low risk

if __name__ == "__main__":
    pytest.main()