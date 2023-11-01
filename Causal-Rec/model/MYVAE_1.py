import torch
import numpy as np
from math import sqrt
from torch import nn
from torch.nn import functional as F
from model.abstracRecommender import GeneralRecommender


class Self_Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, num_heads=4):
        super(Self_Attention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.linear_q = nn.Linear(input_dim, dim_k)
        self.linear_k = nn.Linear(input_dim, dim_k)
        self.linear_v = nn.Linear(input_dim, dim_v)
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / sqrt(dim_k)
        self.num_heads = num_heads
        self.count = 0


    def _expand(self, data):
        '''
        Quantitative Polarization
        @param data:
        @return:
        '''

        return data

    def causal_struc(self, adj, att):
        return torch.mul(adj, att)

    def mutihead(self, x, adj):
        batch, n, in_dim = x.shape
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = self.causal_struc(adj, dist)
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att, dist

    def onhead(self, x, adj):
        Q = self.linear_q(x)  # Q: batch_size * seq_len * dim_k
        K = self.linear_k(x)  # K: batch_size * seq_len * dim_k
        V = self.linear_v(x)  # V: batch_size * seq_len * dim_v

        att_weight = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 2, 1))) * self._norm_fact
        att_weight = self.causal_struc(adj, att_weight)
        att = torch.matmul(att_weight, V)
        return att, att_weight

    def forward(self, c, u, adj, I):
        if u is not None:
            c = torch.add(c, u)  # add recall 下降 b x n x dim
            c = F.normalize(c, dim=-1)  # layernorm? or no norm
        if self.num_heads > 1:
            c, weight = self.mutihead(c, adj)
        else:
            c, weight = self.onhead(c, adj)

        return c, weight


class GraphModel(nn.Module):
    def __init__(self,device = 'cpu'):
        self.device = device
        super(GraphModel, self).__init__()

    def _sample_gumbel(self, shape, eps=1e-20, seed=1228):
        eps = torch.tensor(eps)
        torch.manual_seed(seed)
        u = torch.rand(shape)
        u = -torch.log(-torch.log(u + eps) + eps)
        u[np.arange(shape[0]), np.arange(shape[0])] = 0

        return u.to(self.device)

    def gumbel_sigmoid(self, logits, temperature,seed):

        gumbel_softmax_sample = (logits
                     + self._sample_gumbel(logits.shape, seed=seed)
                     - self._sample_gumbel(logits.shape, seed=seed + 1))
        y = torch.sigmoid(gumbel_softmax_sample / temperature)

        return y

    def sample(self, w, tau, seed=1228):
        '''
        @param w: the causal matrix
        @param tau: the temperature of gumbel
        @param seed:
        @return: the causal matrix with value in 0 or 1.
        '''

        w = self.gumbel_sigmoid(w,tau,seed)
        w = (1. - torch.eye(w.shape[0]).to(self.device)) * w

        return w


class MYVAE(GeneralRecommender):
    def __init__(self,config, dataGeneartor, graph_mask=None):
        super().__init__(config,dataGeneartor)

        # parameters
        self.layers = config.model_parameter.encoder_hidden_size.value #ecoder and decoder mlps channel
        self.lat_dim = config.model_parameter.embeddingsize.value # the dim of latent Z
        self.concepts = config.model_parameter.concepts.value # the number of candiate confounders
        self.concepts_dim = config.model_parameter.concepts_dim.value # the dim of candiate confounders
        self.tau = config.model_parameter.tau.value # the temoreture for gumbal sample
        self.seed = config.model_parameter.seed.value # the random seed
        self.l1_a = config.model_parameter.l1_a.value #
        self.c_A = config.model_parameter.c_A.value #
        self.beta = config.model_parameter.beta.value #
        self.lambda_A = config.model_parameter.lambda_A.value #
        self.concepts_k = config.model_parameter.concepts_k.value # the max number in each Z can be seperate
        # exogenous nodes and endogenous nodes representation on user preference
        self.drop_out = config.model_parameter.dropout.value
        self.anneal_cap = config.model_parameter.anneal_cap.value
        self.total_anneal_steps = config.model_parameter.anneal_steps.value
        self.rol = config.model_parameter.rol.value
        self.rol_muti = config.model_parameter.rol_mutiply.value
        self.alpha = config.model_parameter.alpha.value
        self.h_threshold = config.model_parameter.h_threshold.value
        self.device = config.Device

        # funcs
        self.global_training = True
        self.concepts_emb = nn.Embedding(self.concepts,self.concepts_dim)  # the representation of confounders
        self.build_histroy_items(dataGeneartor)
        self.update = 0
        self.att = Self_Attention(self.concepts_dim, self.concepts_dim * 2, self.concepts_dim, num_heads=self.concepts)
        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]
        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder_x = self.mlp_layers(self.decode_layer_dims)
        self.decoder_c = self.mlp_layers(self.decode_layer_dims)
        self.mixlayer = nn.Linear(self.concepts_dim * 2, self.concepts_dim)
        self.latent_u = nn.Linear(self.lat_dim, self.concepts_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.concepts_dim * 2, self.concepts_dim * 2),
            nn.LayerNorm(self.concepts_dim * 2, eps=0.1),
            nn.ELU(),
            nn.Linear(self.concepts_dim * 2, self.concepts_dim)
        )
        self.trans_z_c = []
        self.trans_z_v = []
        for k in range(self.concepts):
            self.trans_z_c.append(
                nn.Sequential(
                    nn.Linear(self.lat_dim // 2, self.lat_dim),
                    nn.ELU(),
                    nn.Linear(self.lat_dim, self.lat_dim // 2)
                ).to(self.device)
            )
            self.trans_z_v.append(
                nn.Sequential(
                    nn.Linear(self.lat_dim // 2, self.lat_dim),
                    nn.ELU(),
                    nn.Linear(self.lat_dim, self.lat_dim // 2)
                ).to(self.device)
            )
        # parameters initialization
        self.apply(self._xavier_uniform_initialization)
        self.graphmodel = GraphModel(device=self.device)
        a = torch.nn.init.uniform_(torch.Tensor(self.concepts, self.concepts), a=-1e-10, b=1e-10)
        self.A = nn.Parameter(a)  # the causal matrix of total confounders, concepts * concepts
        self.I = nn.Parameter(torch.eye(self.concepts), requires_grad=False)  # eye matrix
        self.h = None
        self.graph_mask = nn.Parameter(graph_mask if graph_mask is not None else torch.ones(self.concepts,self.concepts), requires_grad= False)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.LayerNorm(d_out, eps=0.1))
                mlp_modules.append(nn.ELU())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def _cal_concepts(self, z, var, con_emb):
        z_list = []
        val_list = []
        z_all = None
        for k in range(self.concepts):
            z_ = self.trans_z_c[k](z).unsqueeze(1)
            val_list.append(self.trans_z_v[k](var))
            z_list.append(z_)
            z_all = z_ if z_all is None else torch.cat((z_all, z_), dim=1)
        cates_logits = torch.mul(z_all, con_emb).sum(dim=-1) / self.tau
        # Gumbel-Softmax to generate values like one-hot
        cates_sample = F.gumbel_softmax(cates_logits, tau=0.2, hard=False, dim=-1)
        cates_mode = torch.softmax(cates_logits, dim=-1)
        # cates = self.training * cates_sample + (1 - self.training) * cates_mode  # b x concepts_k x concepts

        # concepts = F.normalize(self.concepts_emb.weight, dim=1)
        # cates_logits = torch.matmul(concept, concepts.transpose(0, 1)) / self.tau
        # cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1) # Gumbel-Softmax to generate values like one-hot
        # cates_mode = torch.softmax(cates_logits, dim=-1)
        # cates = self.training * cates_sample + (1 - self.training) * cates_mode # b x concepts_k x concepts
        return z_list, val_list, z_all, cates_sample

    def matrix_poly(self,matrix,d):
        x = self.I + torch.div(matrix, d)
        return torch.matrix_power(x, d)

    def _h_A(self,adj_A):
        '''
        acyclicity constraint
        @param adj_A: the graph of causal
        @return:
        '''
        expm_A = self.matrix_poly(adj_A * adj_A, self.concepts)
        h_A = torch.trace(expm_A) - self.concepts
        return h_A

    def _cal_adjs(self,adj, inverse = False):
        if not inverse:
            return self.I - (adj.transpose(0, 1))
        else:
            return torch.inverse(self.I - adj.transpose(0, 1))

    def mix(self, u, z):
        return self.mixlayer(torch.cat((u, z), dim=1))

    def get_adj(self, h_threshold=0.5):
        d = self.A.shape[0]
        w_final_weight = self.graphmodel.sample(self.A, self.tau)
        w_final = w_final_weight.clone()  # final graph
        if h_threshold > 0:
            w_final[w_final <= h_threshold] = 0  # Thresholding
            w_final[w_final > h_threshold] = 1
        w_final[np.arange(d), np.arange(d)] = 0  # Mask diagonal to 0
        return w_final

    def mask_graph(self, adj, mask=None):
        if mask is None:
            return torch.mul(self.graph_mask,adj)
        else:
            return torch.mul(mask,adj)

    def forward(self, rating_matrix, mask=None):
        #amplify the value of A and accelerate convergence
        # adj_A = torch.sinh(3. * self.A)
        adj_A = self.graphmodel.sample(self.A, self.tau, self.seed)
        self.mask_graph(adj_A)
        # encoder
        h = F.normalize(rating_matrix)
        h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)  # b x 128
        u = self.latent_u(h).unsqueeze(1)  # b x 1 x k
        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2):]

        # calsal begin
        # get causal structure embedding of counfounders, and personal weight of users.
        # return b x concepts x concepts_dim
        if not self.global_training:
            con_emb, weight = self.att(self.concepts_emb.weight, None, self._cal_adjs(adj_A, inverse=False), self.I)
        else:
            con_emb, weight = self.att(self.concepts_emb.weight, u, self._cal_adjs(adj_A, inverse=False), self.I)
        # calculate the weight of c -> p, using gumbel
        # confounders = torch.reshape(mu, (mu.size(0), self.concepts_k, self.concepts_dim))
        # cons = F.normalize(confounders, dim=2)
        mus, vars, z_all, cates = self._cal_concepts(mu,logvar, con_emb)
        z_d = self.ffn(torch.cat((z_all, con_emb), dim=-1))
        z = torch.matmul(cates.unsqueeze(1), z_d).squeeze()
        # z = self.mix(u, z)
        # cates = torch.matmul(cates,self._cal_adjs(adj_A))
        # c = None
        # for k in range(self.concepts_k):
        #     c_k = torch.mm(cates[:,k,:],con_emb).unsqueeze(1)
        #     c = torch.cat((c,c_k),dim= 1) if c is not None else c_k
        # c = c.reshape(c.size(0),-1)
        #
        # mu_mix = self.mix(mu,c)
        # z_u = self.reparameterize(mu, logvar)
        # z_c = self.reparameterize(c,logvar)
        # mu_c = self.ffn(torch.bmm(cates, con_emb).reshape(mu.size(0), -1))
        z_c = self.reparameterize(z, logvar)
        # z_u = self.reparameterize(mu, logvar)
        x_c = self.decoder_c(z_c)
        # x_u = self.decoder_x(z_u)
        # x = torch.add(x_u, x_c)
        if self.training:
            return x_c, mu, mus, vars, u, logvar, adj_A, weight
        else:
            return x_c, mu, mus, u, logvar

    def calculate_loss(self, interaction):
        user = interaction
        rating_matrix = self.get_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        x, mu, mus, vars, u, logvar, adj_A, att = self.forward(rating_matrix)

        # KL loss
        kl_loss = None
        for k in range(self.concepts):
            kl = torch.mean(torch.sum(1 + vars[k] - mus[k].pow(2) - vars[k].exp(), dim=1))
            kl_loss = kl if kl_loss is None else kl_loss + kl
        # kl_loss = kl_loss + torch.mean(torch.sum(1 + logvar - u.pow(2) - logvar.exp(), dim=1))
        # kl_loss = kl_loss + torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        kl_loss = -0.5 * kl_loss * anneal

        # CE loss
        ce_loss = -(F.log_softmax(x, 1) * rating_matrix).sum(1).mean()

        #h loss
        h_a = self._h_A(adj_A) #+ self._h_A(att)
        self.h = h_a
        # h_loss = self.lambda_A * h_a + 0.5 * self.c_A * h_a * h_a + 100. * torch.trace( adj_A* adj_A) + self.l1_a * torch.abs(adj_A).sum()
        h_loss = self.alpha * h_a + 0.5 * self.rol * h_a * h_a + self.l1_a * torch.linalg.norm(adj_A, ord=1)
        if self.global_training:
            return ce_loss + self.beta * kl_loss + h_loss
        else:
            return ce_loss + kl_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def fullrank(self, interaction):
        user = interaction

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _, _, _ = self.forward(rating_matrix)

        return scores

    def update_rol(self):
        self.rol *= self.rol_muti

    def get_rol(self):
        return self.rol

    def update_alpha(self):
        self.alpha += self.rol * self.h.detach()

    def get_h(self):
        return self.h.detach(), self.h_threshold

    def global_state_change(self):
        self.global_training = ~self.global_training
        if self.global_training:
            self.A.requires_grad = True
            self.concepts_emb.requires_grad_(requires_grad=False)
        else:
            self.A.requires_grad = False
            self.concepts_emb.requires_grad_(requires_grad=True)
