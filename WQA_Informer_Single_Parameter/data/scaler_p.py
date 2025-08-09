import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the training data (same as used in your model)
df = pd.read_csv("CSV file name here") #Also put CSV file in the same directory
df.columns = df.columns.str.strip()

