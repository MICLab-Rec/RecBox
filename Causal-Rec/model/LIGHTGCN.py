import torch
from torch.nn.init import  xavier_uniform_,constant_

class LIGHTGCN(torch.nn.Module):

    def __init__(self, config,datainfo,graph) -> None:
        super(LIGHTGCN, self).__init__()
        self.user_num = datainfo.user_num
        self.item_num = datainfo.item_num
        self.embedim = config.embedding_size
        self.n_layers = 3

        self.user = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedim)
        self.item = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedim)

        torch.nn.init.normal_(self.user.weight, std=0.01)
        torch.nn.init.normal_(self.item.weight, std=0.01)

        self.norm_adj_matrix = graph.to(config.Device)
        self.output = torch.nn.Sigmoid()

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user.weight
        item_embeddings = self.item.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


    def computer(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.user_num, self.item_num]
        )
        return user_all_embeddings, item_all_embeddings

    def forward(self, data, **kwargs):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[data[0]]
        item_emb = all_items[data[1]]
        score = torch.mul(users_emb, item_emb).sum(dim=1)
        return self.output(score)

    def predict(self,data):
        all_users, all_items = self.computer()
        users_emb = all_users[data[0]]
        item_emb = all_items[data[1]]
        score = torch.mul(users_emb, item_emb).sum(dim=1)
        return self.output(score)


    def fullrank(self,data):
        user,mask = data[0],data[1]
        all_users, all_items = self.computer()
        users_emb = all_users[data[0]].unsqueeze(1)
        masked = torch.ones_like(all_items)
        masked[mask] = 0
        item_emb = all_items * masked

        return  self.output(torch.mul(users_emb, item_emb).sum(dim= -1).squeeze() )

