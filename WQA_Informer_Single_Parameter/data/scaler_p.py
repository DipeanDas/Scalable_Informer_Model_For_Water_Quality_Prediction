import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the training data (same as used in your model)
df = pd.read_csv("CSV file name here") #Also put CSV file in the same directory
df.columns = df.columns.str.strip()

# Convert 'Month' to numerical (0–11) and drop original
df['Month'] = pd.to_datetime(df['Month'], format='%m')
df['month'] = df['Month'].dt.month - 1
df.drop(columns=['Month'], inplace=True)

# Encode year relative to base year
df['year'] = df['Year'].astype(int) - int(df['Year'].min())
df.drop(columns=['Year'], inplace=True)

# Ensure Target Column is the last column
target_col = 'Target Column for prediction'
if df.columns[-1] != target_col:
    cols = [col for col in df.columns if col != target_col] + [target_col]
    df = df[cols]


# Drop target column for input feature scaling
input_features = df.drop(columns=[target_col])

# Fit scaler on input features
scaler_input = StandardScaler()
scaler_input.fit(input_features)

# Save the input scaler
joblib.dump(scaler_input, 'scaler_input.pkl')