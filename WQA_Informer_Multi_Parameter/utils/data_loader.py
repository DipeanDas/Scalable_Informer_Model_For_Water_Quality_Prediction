import os
import torch
from torch.utils.data import DataLoader
from utils.custom_data_process import BODDataset
import random

def get_dataloader(
    data_path,
    seq_len=6,
    label_len=3,
    pred_len=1,
    batch_size=16,
    features='M',
    scale=True,
    shuffle=True,
    drop_last=True,
    split_ratio=(0.7, 0.1, 0.2),
    target_cols=None  # <-- This supports multi-targets
):
    """
    Returns train, val, test DataLoaders.
    """

    # Load full dataset
    full_dataset = BODDataset(
        data_path=data_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
        target_cols=target_cols,  # <-- assign multi-target columns
        scale=scale
    )

    total_len = len(full_dataset)
    indices = list(range(total_len))
    random.seed(42)  
    random.shuffle(indices)

    # Apply shuffled indices to dataset
    full_dataset = torch.utils.data.Subset(full_dataset, indices)

    train_end = int(total_len * split_ratio[0])
    val_end = train_end + int(total_len * split_ratio[1])

    train_set = torch.utils.data.Subset(full_dataset, range(0, train_end))
    val_set = torch.utils.data.Subset(full_dataset, range(train_end, val_end))
    test_set = torch.utils.data.Subset(full_dataset, range(val_end, total_len))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, full_dataset
