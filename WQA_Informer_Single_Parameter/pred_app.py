import torch
import pandas as pd
import numpy as np
import joblib
from models.informer import Informer

# Adjust all config values as per requirements
config = {
    'seq_len': 24,
    'label_len': 6,
    'pred_len': 1,
    'enc_in': 25,
    'dec_in': 25,
    'c_out': 1,
    'd_model': 128,
    'n_heads': 4,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'attn': 'prob',
    'embed': 'timeF',
    'freq': 'm',
    'activation': 'gelu',
    'output_attention': False,
    'distil': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

model = Informer(
    enc_in=config['enc_in'], dec_in=config['dec_in'], c_out=config['c_out'],
    seq_len=config['seq_len'], label_len=config['label_len'], pred_len=config['pred_len'],
    factor=5, d_model=config['d_model'], n_heads=config['n_heads'],
    e_layers=config['e_layers'], d_layers=config['d_layers'], d_ff=config['d_ff'],
    dropout=config['dropout'], attn=config['attn'], embed=config['embed'], freq=config['freq'],
    activation=config['activation'], output_attention=config['output_attention'], distil=config['distil']
).to(config['device'])

model.load_state_dict(torch.load(
    "./checkpoints/Ganga-BOD-Test/Ganga-BOD-Test.pth",
    map_location=config['device']
))
model.eval()

scaler_input = joblib.load("scaler_input.pkl")
scaler_bod = joblib.load("scaler_bod.pkl")

data = pd.read_csv("./data/SWD_informers.csv")
data.columns = data.columns.str.strip()
data['Month'] = pd.to_datetime(data['Month'], format='%m')
data['month'] = data['Month'].dt.month - 1
data.drop(columns=['Month'], inplace=True)
base_year = data['Year'].min()
data['year'] = data['Year'] - base_year
data.drop(columns=['Year'], inplace=True)

def predict_bod(location_num, target_year, target_month):
    target_year_offset = target_year - base_year
    df_loc = data[data['Location'] == location_num].sort_values(['year', 'month'])
    if len(df_loc) < config['seq_len']:
        return None

    input_seq = df_loc.iloc[-config['seq_len']:].copy()
    input_features = input_seq.drop(columns=['BOD (mg/l)'])
    scaled_input = scaler_input.transform(input_features)
    seq_x = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(config['device'])

    enc_mark = input_seq[['month', 'year']].values.astype(np.float32)
    dec_inp = torch.zeros(
        (1, config['label_len'] + config['pred_len'], config['dec_in']),
        dtype=torch.float32
    ).to(config['device'])

    dec_mark = np.concatenate([
        enc_mark[-config['label_len']:],
        np.array([[target_month - 1, target_year_offset]], dtype=np.float32)
    ], axis=0)

    enc_mark = torch.tensor(enc_mark, dtype=torch.float32).unsqueeze(0).to(config['device'])
    dec_mark = torch.tensor(dec_mark, dtype=torch.float32).unsqueeze(0).to(config['device'])

    with torch.no_grad():
        output = model(seq_x, enc_mark, dec_inp, dec_mark)

    pred_scaled = output[:, -1:, -1:].cpu().numpy()
    dummy = np.zeros((1, scaler_bod.mean_.shape[0]))
    dummy[0, -1] = pred_scaled[0, 0, 0]
    bod_pred = scaler_bod.inverse_transform(dummy)[0, -1]

    return bod_pred
