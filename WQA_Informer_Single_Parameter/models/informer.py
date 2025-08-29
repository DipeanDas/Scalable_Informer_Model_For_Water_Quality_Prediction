import torch
import torch.nn as nn
from models.embed import DataEmbedding
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import ProbAttention, FullAttention, AttentionLayer

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=128, n_heads=4, e_layers=2, d_layers=1,
                 d_ff=512, dropout=0.0, attn='prob', embed='timeF',
                 freq='m', activation='gelu', output_attention=False, distil=True, mix=True):
        super(Informer, self).__init__()

        self.pred_len = pred_len
        self.output_attention = output_attention

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed_type=embed, dropout=dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed_type=embed, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            conv_layer=ConvLayer(d_model) if distil else None
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, _ = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
