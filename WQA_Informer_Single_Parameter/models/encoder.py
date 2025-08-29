import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(self, x, attn_mask=None):
        # Self-attention + residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(new_x))

        # Feedforward + residual
        residual = x
        x = x.permute(0, 2, 1)
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.norm2(residual + x)

        return x, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, conv_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.conv_layer = conv_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for i, layer in enumerate(self.attn_layers):
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
            if self.conv_layer is not None and i != len(self.attn_layers) - 1:
                x = self.conv_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

    
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in,
                                  kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [Batch, Length, Channel] -> [Batch, Channel, Length]
        x = x.permute(0, 2, 1)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x
