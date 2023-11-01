import torch
import torch.nn as nn
from SeqRec.models.trs_based.trs_layer import TrsLayer, TrsEncoder
from SeqRec.models.trs_based.utils import get_mask


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout_ratio):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        lookup_table = self.pos_embedding.weight[:x.size(1), :]
        x += lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.dropout(x)
        return x


class SASRec(nn.Module):
    def __init__(self, n_item, args):
        super(SASRec, self).__init__()
        self.emb_loc = nn.Embedding(n_item, args.dim, padding_idx=0)
        self.emb_pos = PositionalEmbedding(args.max_len, args.dim, args.dropout_ratio)
        self.trs_layer = TrsLayer(args.dim, args.num_head, args.exp_factor, args.dropout_ratio)
        self.trs_encoder = TrsEncoder(args.dim, self.trs_layer, args.depth)
        self.out = nn.Linear(args.dim, n_item)

    def forward(self, seq, data_size):
        x = self.emb_loc(seq)
        x = self.emb_pos(x)
        mask = get_mask(seq, bidirectional=False)
        output = self.trs_encoder(x, mask)
        if self.training:
            logits = self.out(output)
        else:
            logits = output[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            logits = self.out(logits)
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))