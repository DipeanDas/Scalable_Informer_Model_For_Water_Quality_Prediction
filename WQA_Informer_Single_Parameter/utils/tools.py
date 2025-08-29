import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau

def metric(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    # Regression metrics
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)

    # Correlation metrics
    try:
        plcc, _ = pearsonr(true, pred)
    except:
        plcc = np.nan
    try:
        srcc, _ = spearmanr(true, pred)
    except:
        srcc = np.nan
    try:
        krcc, _ = kendalltau(true, pred)
    except:
        krcc = np.nan

    return mse, mae, rmse, r2, plcc, srcc, krcc

def print_metrics(mse, mae, rmse, r2, plcc, srcc, krcc):
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"KRCC: {krcc:.4f}")

def save_checkpoint(model, save_dir, filename):
    """
    Saving model checkpoint. When a better model (lower validation loss) is found during training, it will be saved automatically.
    """
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
