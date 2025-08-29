import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from utils.tools import metric, save_checkpoint
from utils.data_loader import get_dataloader
from models.informer import Informer

class Exp_Informer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader, self.val_loader, self.test_loader, self.train_dataset = get_dataloader(
            data_path=config['data_path'],
            seq_len=config['seq_len'],
            label_len=config['label_len'],
            pred_len=config['pred_len'],
            batch_size=config['batch_size'],
            target=config['target'],
            features=config['features'],            
            shuffle=config.get('shuffle', True),
            drop_last=config.get('drop_last', True),
            split_ratio=config.get('split_ratio', (0.7, 0.1, 0.2))
        )       

        self.model = Informer(
            enc_in=config['enc_in'],
            dec_in=config['dec_in'],
            c_out=config['c_out'],
            seq_len=config['seq_len'],
            label_len=config['label_len'],
            pred_len=config['pred_len'],
            factor=config['factor'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            e_layers=config['e_layers'],
            d_layers=config['d_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            attn=config['attn'],
            embed=config['embed'],
            freq=config['freq'],
            activation=config['activation'],
            output_attention=config['output_attention'],
            distil=config['distil']
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        self.save_path = os.path.join('./checkpoints', config['model_id'])
        os.makedirs(self.save_path, exist_ok=True)

    def train(self):
        min_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.long().to(self.device)   # convert to long for embeddings
                batch_y_mark = batch_y_mark.long().to(self.device)

                dec_inp = torch.zeros_like(batch_y).float().to(self.device)
                dec_inp[:, :self.config['label_len'], :] = batch_y[:, :self.config['label_len'], :]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.config['pred_len']:, :]
                batch_y = batch_y[:, -self.config['pred_len']:, :]

                # Select only the target column (assumed last column)
                outputs_bod = outputs[..., -1:]
                batch_y_bod = batch_y[..., -1:]

                loss = self.criterion(outputs_bod, batch_y_bod)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            val_loss = self.validate()

            print(f"Epoch {epoch + 1}/{self.config['epochs']} - "
                  f"Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint(self.model, self.save_path, f"{self.config['model_id']}.pth")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.long().to(self.device)
                batch_y_mark = batch_y_mark.long().to(self.device)

                dec_inp = torch.zeros_like(batch_y).float().to(self.device)
                dec_inp[:, :self.config['label_len'], :] = batch_y[:, :self.config['label_len'], :]

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.config['pred_len']:, :]
                batch_y = batch_y[:, -self.config['pred_len']:, :]

                outputs_bod = outputs[..., -1:]
                batch_y_bod = batch_y[..., -1:]

                loss = self.criterion(outputs_bod, batch_y_bod)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{self.config['model_id']}.pth")))
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.long().to(self.device)
                batch_y_mark = batch_y_mark.long().to(self.device)

                dec_inp = torch.zeros_like(batch_y).float().to(self.device)
                dec_inp[:, :self.config['label_len'], :] = batch_y[:, :self.config['label_len'], :]

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.config['pred_len']:, :]
                batch_y = batch_y[:, -self.config['pred_len']:, :]

                outputs_bod = outputs[..., -1:]     # only target column, can change variable name if required
                batch_y_bod = batch_y[..., -1:]     # only target column, can change variable name if required

                preds.append(outputs_bod.cpu().numpy())
                trues.append(batch_y_bod.cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        trues = np.concatenate(trues, axis=0).reshape(-1, 1)

        # === Inverse transform ===
        scaler = joblib.load('scaler_target.pkl')   # fitted with StandardScaler
        total_features = preds.shape[1] + 24     # 24 input features + 1 target = 25[replace 24 with feature count excluding target]

        dummy_pred = np.zeros((preds.shape[0], scaler.scale_.shape[0]))
        dummy_pred[:, -1] = preds[:, 0]
        preds = scaler.inverse_transform(dummy_pred)[:, -1:]

        dummy_true = np.zeros((trues.shape[0], scaler.scale_.shape[0]))
        dummy_true[:, -1] = trues[:, 0]
        trues = scaler.inverse_transform(dummy_true)[:, -1:]

        # === Compute Metrics ===
        mse, mae, rmse, r2, plcc, srcc, krcc = metric(preds, trues)

        print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        print(f"Correlation - PLCC: {plcc:.4f}, SRCC: {srcc:.4f}, KRCC: {krcc:.4f}")

        # === Save Results ===
        from datetime import datetime
        import matplotlib.pyplot as plt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs("results", exist_ok=True)
        os.makedirs("scores", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        pd.DataFrame({
            'Actual Target': np.round(trues.reshape(-1), 4),   #replace target with parameter name
            'Predicted Target': np.round(preds.reshape(-1), 4)
        }).to_csv(f"results/pred_vs_true_{timestamp}.csv", index=False)

        with open(f"scores/metrics_{timestamp}.txt", 'w') as f:
            f.write(f"MSE:  {mse:.4f}\n")
            f.write(f"MAE:  {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2:   {r2:.4f}\n")
            f.write(f"PLCC: {plcc:.4f}\n")
            f.write(f"SRCC: {srcc:.4f}\n")
            f.write(f"KRCC: {krcc:.4f}\n")        

        return preds, trues


