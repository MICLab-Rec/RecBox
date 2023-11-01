import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from model.abstracRecommender import GeneralRecommender

def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret

class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.M = nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()

    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmd(a)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e

class CAUSALVAE(GeneralRecommender):
    def __init__(self,config,dataGeneartor):
        super().__init__(config,dataGeneartor)

        self.layers = config.model_parameter.encoder_hidden_size.value
        self.lat_dim = config.model_parameter.embeddingsize.value
        self.concepts = config.model_parameter.concepts.value
        self.concepts_k = self.lat_dim // (2 * self.concepts)

        # exogenous nodes and endogenous nodes representation on user preference
        self.concepts_emb = nn.Embedding(self.concepts,self.lat_dim // 2)

        self.A = nn.Parameter(torch.zeros(self.concepts,self.concepts))
        self.I = nn.Parameter(torch.eye(self.concepts),requires_grad= False)
        self.att = Attention(self.concepts_k)
        self.drop_out = config.model_parameter.dropout.value
        self.anneal_cap = config.model_parameter.anneal_cap.value
        self.total_anneal_steps = config.model_parameter.anneal_steps.value

        self.build_histroy_items(dataGeneartor)

        self.update = 0

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [self.concepts_k,int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = nn.Sequential(*[self.mlp_layers(self.decode_layer_dims) for _ in range(self.concepts)])
        self.scale = np.array([[0,1] * self.concepts])
        self.enc = [self.n_items] + self.layers + [self.lat_dim]
        self.mix = self.ffnlayers()

        # parameters initialization
        self.apply(self._xavier_uniform_initialization)

    def mask(self,x):
        x = torch.matmul(self.A.t(), x)
        return x

    def causal_z(self,x):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), bias= None)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x

    def kl_normal(self,mu, logvar):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension
        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance
        Return:
            kl: tensor: (batch,): kl between each sample
        """
        kl_loss = (
                -0.5
                * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        )
        # print("log var1", qv)
        return kl_loss

    def matrix_poly(self,matrix, d):
        x = torch.eye(d).to(self.device) + torch.div(matrix, d)
        return torch.matrix_power(x, d)

    def _h_A(self):
        expm_A = self.matrix_poly(self.A * self.A, self.concepts)
        h_A = torch.trace(expm_A) -  self.concepts
        return h_A

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.ELU())
        return nn.Sequential(*mlp_modules)

    def sep_and_dec(self,x):
        s_x = torch.split(x,self.concepts,dim= 1)
        back = None
        for k,x_ in enumerate(s_x):
            back = back + self.decoder[k - 1](x_) if back is not None else self.decoder[k - 1](x_)
        return back.mean(1)

    def ffnlayers(self):
        '''
        through a FFN layer to enhance the representation of z
        @return:
        '''
        return nn.Sequential(
            nn.Linear(self.concepts_k, 32),
            nn.ELU(),
            nn.Linear(32, self.concepts_k)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = self.encoder(h)

        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2) :]

        # z = self.reparameterize(mu, logvar)
        # rec = self.decoder(z)
        return h, mu, logvar

    def calculate_loss(self, interaction):
        rating_matrix = self.get_rating_matrix(interaction)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        h, q_m, q_v = self.forward(rating_matrix) # b * lat_dim / 2
        q_m = q_m.reshape(q_m.size(0),self.concepts,self.concepts_k) # b * concepts * concepts_k
        causal_z = self.causal_z(q_m)# b * lat_dim / 2
        m_zm = self.mask(causal_z)# b * lat_dim / 2 * 1
        # m_u = self.mask(label)
        f_z = self.mix(m_zm).squeeze()# b * lat_dim / 2, pass a ffn network, improver the representation
        e_tilde = self.att.attention(causal_z,q_m)
        con_z = f_z + e_tilde
        con_z = self.reparameterize(con_z,torch.ones_like(con_z).to(self.device))
        rec = self.sep_and_dec(con_z)

        ce_loss = -(F.log_softmax(rec, 1) * rating_matrix).sum(1).mean()

        p_m, p_v = torch.zeros_like(q_m), torch.ones_like(q_m)
        cp_m, cp_v = torch.zeros_like(causal_z).to(self.device), torch.ones_like(causal_z).to(self.device) #ut.condition_prior(self.scale, label, self.z2_dim)
        #
        # cp_v = torch.ones([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)
        # cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        # kl = torch.zeros(1).to(device)
        kl = 0.3 * self.kl_normal(q_m.view(-1, self.lat_dim // 2), q_v.view(-1, self.lat_dim // 2)) # normal KL
        mask_kl = None
        for i in range(self.concepts):
            kl = kl + 1 * self.kl_normal(causal_z[:, i, :], cp_v[:, i, :])
            mask_kl = mask_kl + 1 * self.kl_normal(con_z[:, i, :], cp_v[:, i, :]) if mask_kl is not None else 1 * self.kl_normal(con_z[:, i, :], cp_v[:, i, :])

        h_a = self._h_A()
        h_a_loss = 3*h_a + 0.5*h_a*h_a

        return ce_loss + kl * anneal + mask_kl.mean() + h_a_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]

        rating_matrix = self.get_rating_matrix(user)

        h, q_m, q_v = self.forward(rating_matrix)  # b * lat_dim / 2
        q_m = q_m.reshape(q_m.size(0), self.concepts, self.concepts_k)  # b * concepts * concepts_k
        causal_z = self.causal_z(q_m)  # b * lat_dim / 2
        m_zm = self.mask(causal_z)  # b * lat_dim / 2 * 1
        # m_u = self.mask(label)
        f_z = self.mix(m_zm).squeeze()  # b * lat_dim / 2, pass a ffn network, improver the representation
        e_tilde = self.att.attention(causal_z, q_m)
        con_z = f_z + e_tilde
        con_z = self.reparameterize(con_z, torch.ones_like(con_z).to(self.device))
        scores = self.sep_and_dec(con_z)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def fullrank(self, interaction):
        user = interaction

        rating_matrix = self.get_rating_matrix(user)

        h, q_m, q_v = self.forward(rating_matrix)  # b * lat_dim / 2
        q_m = q_m.reshape(q_m.size(0), self.concepts, self.concepts_k)  # b * concepts * concepts_k
        causal_z = self.causal_z(q_m)  # b * lat_dim / 2
        m_zm = self.mask(causal_z)  # b * lat_dim / 2 * 1
        # m_u = self.mask(label)
        f_z = self.mix(m_zm).squeeze()  # b * lat_dim / 2, pass a ffn network, improver the representation
        e_tilde = self.att.attention(causal_z, q_m)
        con_z = f_z + e_tilde
        con_z = self.reparameterize(con_z, torch.ones_like(con_z).to(self.device))
        scores = self.sep_and_dec(con_z)

        return scores

