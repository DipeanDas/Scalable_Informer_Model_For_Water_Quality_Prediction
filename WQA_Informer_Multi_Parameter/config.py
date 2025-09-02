# config.py

def get_config():
    return {
        'model_id': 'Put test ID here',
        'data_path': 'path to csv file',

        # Sequence lengths
        'seq_len': 24,      # past time steps used as input
        'label_len': 6,     # part of decoder input
        'pred_len': 1,      # how far to predict (1 month)

        # Model dimensions
        'enc_in': 23,       # 26 total cols - 3 targets = 23 input features; update as per csv file 
        'dec_in': 23,
        'c_out': 3,         # predicting 3 targets ;  update as per target field count

        # Training parameters
        'batch_size': 16,
        'epochs': 20,
        'lr': 0.0005,

        # Informer-specific
        'factor': 5,
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
        'features': 'M',
        'target_cols': ['target1', 'target2', 'target_n'], # update as per target fields 
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'use_gpu': True,
        'num_workers': 0,
        'shuffle': True,
    }
