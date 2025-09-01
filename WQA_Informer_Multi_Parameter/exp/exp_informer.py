import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from utils.tools import metric, print_metrics, save_checkpoint
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
            target_cols=config['target_cols'],
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
                batch_x_mark = batch_x_mark.to(self.device).long()
                batch_y_mark = batch_y_mark.to(self.device).long()

                dec_inp = torch.zeros((batch_y.shape[0], self.config['label_len'] + self.config['pred_len'], self.config['dec_in'])).float().to(self.device)
                dec_inp = dec_inp.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.config['pred_len']:, :]
                batch_y = batch_y[:, -self.config['pred_len']:, :]

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.config['epochs']} - Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")

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
                batch_x_mark = batch_x_mark.to(self.device).long()
                batch_y_mark = batch_y_mark.to(self.device).long()

                dec_inp = torch.zeros((batch_y.shape[0], self.config['label_len'] + self.config['pred_len'], self.config['dec_in'])).float().to(self.device)
                dec_inp = dec_inp.to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.config['pred_len']:, :]
                batch_y = batch_y[:, -self.config['pred_len']:, :]

                loss = self.criterion(outputs, batch_y)
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
                batch_x_mark = batch_x_mark.to(self.device).long()
                batch_y_mark = batch_y_mark.to(self.device).long()

                dec_inp = torch.zeros((batch_y.shape[0], self.config['label_len'] + self.config['pred_len'], self.config['dec_in'])).float().to(self.device)
                dec_inp = dec_inp.to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.config['pred_len']:, :]
                batch_y = batch_y[:, -self.config['pred_len']:, :]

                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1, self.config['c_out'])
        trues = np.concatenate(trues, axis=0).reshape(-1, self.config['c_out'])
       
        scaler = joblib.load('scaler_target.pkl')
        col_order = joblib.load('column_order_target.pkl')

        # Finding correct target column indices
        target_indices = [col_order.index(col) for col in self.config['target_cols']]

        dummy_pred = np.zeros((preds.shape[0], len(col_order)))
        dummy_true = np.zeros((trues.shape[0], len(col_order)))

        # Placing each target prediction back in correct scaler column index
        for i, idx in enumerate(target_indices):
            dummy_pred[:, idx] = preds[:, i]
            dummy_true[:, idx] = trues[:, i]

        # Inversing transform the full dummy arrays
        preds_inv = scaler.inverse_transform(dummy_pred)[:, target_indices]
        trues_inv = scaler.inverse_transform(dummy_true)[:, target_indices]


        # Evaluation Phase
        results = metric(preds_inv, trues_inv)
        target_cols = self.config['target_cols']
        print_metrics(results, target_cols)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')        
        
        # Save metrics to text file        
        os.makedirs("scores", exist_ok=True)
        with open(f"scores/metrics_{timestamp}.txt", "w") as f:
            target_cols = self.config['target_cols']
            for i, target in enumerate(target_cols):
                f.write(f"Metrics for {target}:\n")
                f.write(f"  MSE:  {results[0][i]:.4f}\n")
                f.write(f"  MAE:  {results[1][i]:.4f}\n")
                f.write(f"  RMSE: {results[2][i]:.4f}\n")
                f.write(f"  R²:   {results[3][i]:.4f}\n")
                f.write(f"  PLCC: {results[4][i]:.4f}\n")
                f.write(f"  SRCC: {results[5][i]:.4f}\n")
                f.write(f"  KRCC: {results[6][i]:.4f}\n\n")        
        
        return preds_inv, trues_inv

