import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention
        x_res, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = self.norm1(x + self.dropout(x_res))

        # Cross-attention
        x_res, attn = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = self.norm2(x + self.dropout(x_res))

        # Feedforward
        y = x.permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        x = x + y.permute(0, 2, 1)

        return self.norm3(x), attn

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection  # Final linear layer: d_model -> c_out 

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)  # Apply final projection for multivariate prediction
        return x, attns
