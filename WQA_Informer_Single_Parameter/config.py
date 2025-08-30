# config.py

def get_config():
    return {
        'model_id': 'Put Test ID name',
        'data_path': 'share data path to csv file',
        'target': 'put target column',

        # Sequence lengths
        'seq_len': 24,      # past time steps used as input
        'label_len': 6,     # part of decoder input
        'pred_len': 1,      # how far to predict (1 month)

        # Model dimensions
        'enc_in': 25,       # number of input features (excluding target) # change according to parameter count
        'dec_in': 25,
        'c_out': 1,         # predicting 1 target (BOD)

        # Training parameters  # Can be changed for fine tuning
        'batch_size': 16,
        'epochs': 10,
        'lr': 0.0005,

        # Informer-specific #Also can be changed for better results
        'factor': 5,        # used in attention distillation (commonly 3, 5, or 7)
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

        # Others
        'features': 'M',    # Multivariate
        'target_idx': 'index of target column' ,   # put index value of target column
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'use_gpu': True,
        'num_workers': 0,
        'shuffle': True,
    }
