import torch.nn as nn
import torch

class MF(nn.Module):
    def __init__(self,config,info):
        super(MF, self).__init__()
        self.user_num = info.user_num
        self.item_num = info.item_num
        self.embedding_size = config.embedding_size
        self.user_emb = nn.Embedding(self.user_num, self.embedding_size)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.item_emb = nn.Embedding(self.item_num, self.embedding_size)
        self.item_bias = nn.Embedding(self.item_num, 1)

        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.output_func = nn.Sigmoid()

    def forward(self, data):
        u_id = data[0]
        i_id = data[1]
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return self.output_func(torch.mul(U, I).sum(1) + b_u + b_i + self.mean)

    def predict(self,data):
        u_id = data[0]
        i_id = data[1]
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return self.output_func(torch.mul(U, I).sum(1) + b_u + b_i + self.mean)

    def fullrank(self,data):
        user,mask = data[0],data[1]
        user_emb = self.user_emb(user).unsqueeze(1)
        item_emb = self.item_emb.weight
        # masked = torch.ones_like(item_emb)
        # masked[mask] = 0
        # item_emb = item_emb * masked
        b_u = self.user_bias(user)
        b_i = self.item_bias.weight.squeeze()

        return  self.output_func(torch.mul(user_emb , item_emb).sum(-1).squeeze() + b_u + b_i + self.mean)

