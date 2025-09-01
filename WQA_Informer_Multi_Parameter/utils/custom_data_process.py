import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import joblib

class BODDataset(Dataset):
    def __init__(self, data_path, seq_len, label_len, pred_len, features='M', 
                 target_cols=['target1', 'target2', 'target_n'], # update as per target fields
                 scale=True, timeenc=0, freq='M'):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target_cols = target_cols
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.data = pd.read_csv(data_path)
        self.data.columns = self.data.columns.str.strip()

        # Extracting time features
        self.data['Month'] = pd.to_datetime(self.data['Month'], format='%m')    #Ensure month and year field present in csv
        self.data['month'] = self.data['Month'].dt.month - 1
        self.data.drop(columns=['Month'], inplace=True)
        if 'Year' not in self.data.columns:
            raise ValueError("CSV must contain a 'Year' column")
        self.data['year'] = self.data['Year'].astype(int) - int(self.data['Year'].min())
        self.data.drop(columns=['Year'], inplace=True)        

        self.scaler = StandardScaler()
        self._build_data()

    def _build_data(self):
        numeric_data = self.data.select_dtypes(include=[np.number]).copy()

        if self.scale:
            self.scaler.fit(numeric_data.values)
            joblib.dump(self.scaler, 'scaler_target.pkl')     #replace target with appropriate name if needed
            numeric_data[:] = self.scaler.transform(numeric_data.values)
            self.col_order = numeric_data.columns.tolist()
            joblib.dump(self.col_order, 'column_order_target.pkl')  # important to save column order for inverse transform

        self.data_x = numeric_data
        self.data_y = numeric_data[self.target_cols]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        if r_end > len(self.data_x):
            raise IndexError("Index out of bounds for sequence windowing.")

        # Drop target columns and keep only input features
        input_features = self.data_x.drop(columns=self.target_cols)

        seq_x = input_features.iloc[s_begin:s_end].values
        seq_y = self.data_y.iloc[r_begin:r_end].values

        # Time features for x_mark/y_mark
        time_x = self.data[['month', 'year']].iloc[s_begin:s_end].values.astype(np.float32)
        time_y = self.data[['month', 'year']].iloc[r_begin:r_end].values.astype(np.float32)

        return (
            torch.tensor(seq_x, dtype=torch.float),      # shape: [seq_len, input_size]
            torch.tensor(seq_y, dtype=torch.float),      # shape: [label_len + pred_len, output_size]
            torch.tensor(time_x, dtype=torch.long),      # shape: [seq_len, 2]
            torch.tensor(time_y, dtype=torch.long)       # shape: [label_len + pred_len, 2]
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
