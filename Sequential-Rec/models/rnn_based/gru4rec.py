import torch
import torch.nn as nn


class RNNMixer(nn.Module):
    def __init__(self, d_model, dropout_ratio, depth):
        super(RNNMixer, self).__init__()
        self.rnn_mixer = nn.GRU(input_size=d_model,
                                hidden_size=d_model,
                                num_layers=depth,
                                bias=False,
                                batch_first=True,
                                dropout=dropout_ratio)
        self.h_0 = nn.Parameter(torch.randn((depth, 1, d_model), requires_grad=True))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x, _ = self.rnn_mixer(x, self.h_0.expand(-1, x.size(0), -1).contiguous())
        x = self.norm(x)
        return x


class GRU4Rec(nn.Module):
    def __init__(self, n_loc, d_model=64, dropout_ratio=0.5, depth=2):
        super(GRU4Rec, self).__init__()
        self.emb_loc = nn.Embedding(n_loc, d_model, padding_idx=0)
        self.mixer = RNNMixer(d_model, dropout_ratio, depth)
        self.out = nn.Linear(d_model, n_loc)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, seq, data_size):
        x = self.emb_loc(seq)
        x = self.dropout(x)
        mixer_output = self.mixer(x)
        if self.training:
            output = self.out(mixer_output)
        else:
            output = mixer_output[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            output = self.out(output)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))