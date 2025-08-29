import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import joblib

class BODDataset(Dataset):
    def __init__(self, data_path, seq_len, label_len, pred_len, features='M', target='target column', scale=True, timeenc=0, freq='M'):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.data = pd.read_csv(data_path)
        self.data.columns = self.data.columns.str.strip()
        
        # Extractng time features
        self.data['Month'] = pd.to_datetime(self.data['Month'], format='%m')
        self.data['month'] = self.data['Month'].dt.month-1
        self.data.drop(columns=['Month'], inplace=True)
        if 'Year' not in self.data.columns:
            raise ValueError("CSV must contain a 'Year' column")
        self.data['year'] = self.data['Year'].astype(int) - int(self.data['Year'].min())
        self.data.drop(columns=['Year'], inplace=True)

        # Target must be the last column in the dataset
        if self.target != self.data.columns[-1]:
            cols = [col for col in self.data.columns if col != self.target] + [self.target]
            self.data = self.data[cols]

        self.scaler = StandardScaler()
        self._build_data()

    def _build_data(self):
        # Selecting numeric columns only
        numeric_data = self.data.select_dtypes(include=[np.number]).copy()
        
        if self.scale:
            self.scaler.fit(numeric_data.values)
            joblib.dump(self.scaler, 'scaler_target.pkl')  #can replace the word target with the parameter name for  convenience
            numeric_data[:] = self.scaler.transform(numeric_data.values)

        self.data_x = numeric_data
        self.data_y = numeric_data[[self.target]]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        if r_end > len(self.data_x):
            raise IndexError("Index out of bounds for sequence windowing.")

        # Drop target and keep only input features (should be total input feature count -1)
        input_features = self.data_x.drop(columns=['Target column'])

        seq_x = input_features.iloc[s_begin:s_end].values
        seq_y = input_features.iloc[r_begin:r_end].values 

        # Time features for x_mark/y_mark
        time_x = self.data[['month', 'year']].iloc[s_begin:s_end].values.astype(np.float32)
        time_y = self.data[['month', 'year']].iloc[r_begin:r_end].values.astype(np.float32)
        
        return (
            torch.tensor(seq_x, dtype=torch.float),      # shape: [seq_len, (total input feature count -1)]
            torch.tensor(seq_y, dtype=torch.float),      # shape: [label_len + pred_len, (total input feature count -1)]
            torch.tensor(time_x, dtype=torch.long),     # shape: [seq_len, 2]  [2=month and year]
            torch.tensor(time_y, dtype=torch.long)      # shape: [label_len + pred_len, 2]
        )
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # Undo standard scaling for predictions
        return self.scaler.inverse_transform(data)
    
    #The CSV must have Month and year column, the code is compatible  for months in 1-12 range and Year in double digit type

