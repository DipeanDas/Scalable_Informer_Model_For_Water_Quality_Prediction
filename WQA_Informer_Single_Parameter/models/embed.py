import torch
import torch.nn as nn
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        

    def forward(self, x):
        return 
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        

     # [B, L, D]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF'):
        super().__init__()
        

    

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', dropout=0.1):
        super().__init__()
        

    
