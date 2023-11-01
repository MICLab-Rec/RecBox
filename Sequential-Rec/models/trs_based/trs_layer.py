import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from SeqRec.models.trs_based.utils import PreNorm


def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MHAttn(nn.Module):
    def __init__(self, n_head, d_model, dropout_ratio):
        super(MHAttn, self).__init__()
        self.d_k = d_model // n_head
        self.h = n_head
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        q, k, v = [l(x).view(b, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        x = self.merge(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, exp_factor, dropout):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_model * exp_factor)
        self.linear_2 = nn.Linear(d_model * exp_factor, d_model)
        # self.act = nn.GELU()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class TrsLayer(nn.Module):
    def __init__(self, dim, n_head, exp_factor, dropout_ratio):
        super().__init__()
        self.attn_layer = MHAttn(n_head, dim, dropout_ratio)
        self.ffn_layer = FFN(dim, exp_factor, dropout_ratio)
        self.sublayer_1 = PreNorm(dim)
        self.sublayer_2 = PreNorm(dim)

    def forward(self, x, mask):
        x = self.sublayer_1(x, lambda e: self.attn_layer(e, e, e, mask))
        x = self.sublayer_2(x, self.ffn_layer)
        return x


class TrsEncoder(nn.Module):
    def __init__(self, dim, layer, depth):
        super().__init__()
        self.encoder = nn.ModuleList(
            [layer for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        for layer in self.encoder:
            x = layer(x, mask)
        x = self.norm(x)
        return x
