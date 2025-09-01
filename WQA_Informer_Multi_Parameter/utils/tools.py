import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau

def metric(pred, true):
    """
    Compute evaluation metrics for each target column independently.
    Input shape: [num_samples, num_targets]
    """
    num_targets = pred.shape[1]
    
    results = []
    for i in range(num_targets):
        p = pred[:, i]
        t = true[:, i]

        mse = mean_squared_error(t, p)
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mse)
        r2 = r2_score(t, p)

        try:
            plcc, _ = pearsonr(t, p)
        except:
            plcc = np.nan
        try:
            srcc, _ = spearmanr(t, p)
        except:
            srcc = np.nan
        try:
            krcc, _ = kendalltau(t, p)
        except:
            krcc = np.nan

        results.append([mse, mae, rmse, r2, plcc, srcc, krcc])
    
    return results  # List of [mse, mae, rmse, r2, plcc, srcc, krcc] per target


def print_metrics(results, target_cols=None):
    """
    Pretty print evaluation metrics for each target.
    """
    if target_cols is None:
        target_cols = [f'Target-{i}' for i in range(len(results))]
    
    for i, col in enumerate(target_cols):
        mse, mae, rmse, r2, plcc, srcc, krcc = results[i]
        print(f"\nMetrics for {col}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  PLCC: {plcc:.4f}")
        print(f"  SRCC: {srcc:.4f}")
        print(f"  KRCC: {krcc:.4f}")

def save_checkpoint(model, save_dir, filename):
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
