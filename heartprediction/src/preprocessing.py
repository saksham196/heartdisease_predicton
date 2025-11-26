import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(data):
    # Convert categorical variables to dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Ensure all expected columns are present
    expected_columns = ['Age', 'RestingBP', 'Cholestrol', 'FastingBs', 'MaxHR', 'Oldpeak', 
                        'Sex_M', 'ChestPainType_NAP', 'ChestPainType_TA', 
                        'ChestPainType_ASY', 'RestingECG_ST', 'RestingECG_LVH', 
                        'ExerciseAngina_Y', 'ST_slope_Fast', 'ST_slope_Down']
    
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0
            
    data = data[expected_columns]
    
    return data

def scale_features(data, scaler):
    return scaler.transform(data)