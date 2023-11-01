import copy
import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class FilterLayer(nn.Module):
    def __init__(self, d_model, max_len):
        super(FilterLayer, self).__init__()
        self.filter = nn.Parameter(torch.randn(1, max_len // 2 + 1, d_model, 2, dtype=torch.float32) * 0.02)
        self.drop = nn.Dropout(0.5)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        b, n, d = x.shape
        y = torch.fft.rfft(x, dim=1, norm='ortho')
        kernel = torch.view_as_complex(self.filter)
        y = y * kernel
        y = torch.fft.irfft(y, n=n, dim=1, norm='ortho')
        y = self.drop(y)
        y = self.norm(y + x)
        return y


class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(0.5)
        self.act = nn.GELU()

    def forward(self, x):
        y = gelu(self.linear_1(x))
        y = self.drop(self.linear_2(y))
        y = self.norm(y + x)
        return y


class BasicLayer(nn.Module):
    def __init__(self, d_model, max_len):
        super(BasicLayer, self).__init__()
        self.layer_1 = FilterLayer(d_model, max_len)
        self.layer_2 = FFN(d_model)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y


class Encoder(nn.Module):
    def __init__(self, d_model, max_len, depth):
        super(Encoder, self).__init__()
        self.blk = BasicLayer(d_model, max_len)
        self.encoder = nn.ModuleList([copy.deepcopy(self.blk) for _ in range(depth)])

    def forward(self, x):
        for blk in self.encoder:
            x = blk(x)
        return x


class FMLP4Rec(nn.Module):
    def __init__(self, n_loc, d_model=64, max_len=64, depth=2):
        super(FMLP4Rec, self).__init__()
        self.emb_loc = nn.Embedding(n_loc, d_model, padding_idx=0)
        self.emb_pos = nn.Embedding(max_len, d_model)
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(0.5)
        self.encoder = Encoder(d_model, max_len, depth)
        self.out = nn.Linear(d_model, n_loc)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position(self, src_locs):
        seq_len = src_locs.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=src_locs.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(src_locs)
        loc_embedding = self.emb_loc(src_locs)
        pos_embedding = self.emb_pos(pos_ids)
        x = self.drop(self.norm(loc_embedding + pos_embedding))
        return x

    def forward(self, src_locs, data_size):
        x = self.add_position(src_locs)
        x = self.encoder(x)
        if self.training:
            output = self.out(x)
        else:
            output = x[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            output = self.out(output)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


