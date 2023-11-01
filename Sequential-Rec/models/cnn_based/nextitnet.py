import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    def __init__(self, input_features, output_features, kernel_size, dilation, dropout):
        super(CNNLayer, self).__init__()
        self.conv_1 = nn.Conv2d(input_features, output_features, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.norm_1 = nn.LayerNorm(output_features)
        self.conv_2 = nn.Conv2d(output_features, output_features, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        self.norm_2 = nn.LayerNorm(output_features)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def conv_pad(self, x, dilation):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2)
        pad_function = nn.ZeroPad2d(((self.kernel_size-1) * dilation, 0, 0, 0))
        x = pad_function(x)
        return x

    def forward(self, x):
        x_pad = self.conv_pad(x, self.dilation)
        y = self.conv_1(x_pad).squeeze(2).permute(0, 2, 1)
        y = self.act(self.norm_1(y))
        y_pad = self.conv_pad(y, self.dilation*2)
        y = self.conv_2(y_pad).squeeze(2).permute(0, 2, 1)
        y = self.act(self.norm_2(y))
        return self.dropout(y) + x


class CNNMixer(nn.Module):
    def __init__(self, input_features, output_features, kernel_size, dilations, dropout):
        super(CNNMixer, self).__init__()
        self.dilations = dilations
        self.layer = [CNNLayer(input_features, output_features, kernel_size, dilation=dilation, dropout=dropout)
                      for dilation in self.dilations]
        self.mixer = nn.Sequential(*self.layer)

    def forward(self, x):
        x = self.mixer(x)
        return x


class NextItNet(nn.Module):
    def __init__(self, n_loc, d_model=64, dropout=0.5, depth=2):
        super(NextItNet, self).__init__()
        self.emb_loc = nn.Embedding(n_loc, d_model, padding_idx=0)
        self.dilation = [1, 4] * depth
        self.kernel_size = 3
        self.mixer = CNNMixer(d_model, d_model, self.kernel_size, self.dilation, dropout)
        self.out = nn.Linear(d_model, n_loc)
        self.dropout = nn.Dropout(dropout)

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