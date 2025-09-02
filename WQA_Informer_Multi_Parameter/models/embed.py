import torch
import torch.nn as nn
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].clone().detach()

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: [Batch, Seq Len, Features]
        
        x = x.permute(0, 2, 1)  # [B, C, L]
        
        x = self.tokenConv(x)
        x = self.activation(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [B, L, D]
        
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF'):
        super().__init__()
        self.month_embed = nn.Embedding(13, d_model)  # months 0–11 + pad (12)
        self.year_embed = nn.Embedding(20, d_model)   # Support up to 20 years (increase as per requirement)
        self.embed_type = embed_type

    def forward(self, x):
        # x: [Batch, Seq Len, 2] → [month, year]
        month_x = self.month_embed(x[:, :, 0].long())
        year_x = self.year_embed(x[:, :, 1].long())
        return month_x + year_x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [B, L, C]; x_mark: [B, L, T]
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
