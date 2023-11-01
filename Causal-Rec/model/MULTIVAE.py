from torch import nn
import torch
from torch.nn import  functional as F
from model.abstracRecommender import  GeneralRecommender


class MULTIVAE(GeneralRecommender):
    r"""MultiVAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

    We implement the MultiVAE model with only user dataloader.
    """

    def __init__(self, config, dataGenerator):
        super(MULTIVAE, self).__init__(config,dataGenerator)

        self.layers = config.model_parameter.layers.value
        self.lat_dim = config.model_parameter.embeddingsize.value
        self.drop_out = config.model_parameter.dropout.value
        self.anneal_cap = config.model_parameter.anneal_cap.value
        self.total_anneal_steps = config.model_parameter.anneal_steps.value

        self.build_histroy_items(dataGenerator)

        self.update = 0

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][
            1:
        ]

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

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar

    def calculate_loss(self, interaction):
        user = interaction
        rating_matrix = self.get_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = self.forward(rating_matrix)

        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        return ce_loss + kl_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def fullrank(self, interaction):
        user = interaction

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores