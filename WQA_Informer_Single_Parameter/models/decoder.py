import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention
        x_res, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(x_res)
        x = self.norm1(x)      

        

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    
