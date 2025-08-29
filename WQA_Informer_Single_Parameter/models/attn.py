import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        scores = torch.einsum("bnqd,bnkd->bnqk", queries, keys)

        if self.scale:
            scores = scores / self.scale
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L_Q, L_K), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        A = self.dropout(F.softmax(scores, dim=-1))
        V = torch.matmul(A, values)

        if self.output_attention:
            return V, A
        else:
            return V, None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.scale = scale
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k)).to(K.device)  # uniform sampling
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # [B, H, L_Q, sample_k, D]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [B, H, L_Q, sample_k]

        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)  # [B, H, L_Q]
        
        
        n_top_safe = min(n_top, L_Q)
        M_top = M.topk(n_top_safe, sorted=False)[1]  # [B, H, n_top_safe]

        # Ensuring Q_reduce is correctly indexed for batched inputs
        batch_indices = torch.arange(B)[:, None, None].to(Q.device)
        head_indices = torch.arange(H)[None, :, None].to(Q.device)
        
        Q_reduce = Q[batch_indices, head_indices, M_top, :]  # [B, H, n_top_safe, D]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # [B, H, n_top_safe, L_K]
        
        return Q_K, M_top

    def forward(self, queries, keys, values, attn_mask):
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        queries = queries / math.sqrt(D)

        U_part = self.factor * math.ceil(math.log(L_K))
        u = self.factor * math.ceil(math.log(L_Q))

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scores = torch.matmul(queries, keys.transpose(-2, -1))

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L_Q, L_K), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        A = self.dropout(F.softmax(scores, dim=-1))
        V = torch.matmul(A, values)

        if self.output_attention:
            return V, A
        else:
            return V, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape  # keys length can differ
        H = self.n_heads
        D_k = self.d_keys
        D_v = self.d_values

        queries = self.query_projection(queries).view(B, L_Q, H, D_k).transpose(1, 2)  # [B, H, L_Q, D_k]
        keys = self.key_projection(keys).view(B, L_K, H, D_k).transpose(1, 2)          # [B, H, L_K, D_k]
        values = self.value_projection(values).view(B, L_K, H, D_v).transpose(1, 2)    # [B, H, L_K, D_v]

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, L_Q, -1)
        return self.out_projection(out), attn


