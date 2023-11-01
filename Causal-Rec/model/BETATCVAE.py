from torch import nn
import torch
from torch.nn import  functional as F
from model.abstracRecommender import  GeneralRecommender
from utils.distribution import Bernoulli,Normal
from torch.autograd import Variable

from utils.functions import log_density_gaussian, matrix_log_density_gaussian, log_importance_weight_matrix


class BETATCVAE(GeneralRecommender):
    '''

    '''

    def __init__(self, config, dataGenerator):
        super(BETATCVAE, self).__init__(config,dataGenerator)

        self.layers = config.model_parameter.layers.value
        self.lat_dim = config.model_parameter.embeddingsize.value
        self.drop_out = config.model_parameter.dropout.value
        self.anneal_cap = config.model_parameter.anneal_cap.value
        self.total_anneal_steps = config.model_parameter.anneal_steps.value
        self.beta = config.model_parameter.beta.value
        self.gamma = config.model_parameter.gamma.value
        self.alpha = config.model_parameter.alpha.value
        self.register_buffer('prior_params', torch.zeros(self.lat_dim, 2))
        self.data_size = dataGenerator.datainfo.user_num

        self.build_histroy_items(dataGenerator)

        self.update = 0

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)

        # parameters initialization
        self.apply(self._xavier_uniform_initialization)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    def _get_log_pz_qz_prodzi_qzCx(self,latent_sample, mu,logvar, n_data, is_mss=True):
        batch_size, hidden_dim = latent_sample.shape

        # calculate log q(z|x)
        log_q_zCx = log_density_gaussian(latent_sample, mu,logvar).sum(dim=1)

        # calculate log p(z)
        # mean and log var is 0
        zeros = torch.zeros_like(latent_sample)
        log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

        mat_log_qz = matrix_log_density_gaussian(latent_sample, mu,logvar)

        if is_mss:
            # use stratification
            log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
            mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

        log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

        return log_pz, log_qz, log_prod_qzi, log_q_zCx

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

        z_s = self.reparameterize(mu, logvar)
        x = self.decoder(z_s)
        return x,z_s,mu, logvar

    def calculate_loss(self, interaction):
        user = interaction
        rating_matrix = self.get_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        x,z_s,mu, logvar = self.forward(rating_matrix)

        # CE Loss
        ce_loss = -(F.log_softmax(x, 1) * rating_matrix).sum(1).mean()

        log_pz, log_qz, log_prod_qzi, log_q_zCx = self._get_log_pz_qz_prodzi_qzCx(z_s,mu,logvar,self.data_size)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        # total loss
        loss = ce_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal * self.gamma * dw_kl_loss)

        return loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _,_ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def fullrank(self, interaction):
        user = interaction

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _,_ = self.forward(rating_matrix)

        return scores