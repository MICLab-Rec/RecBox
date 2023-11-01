from torch import nn
import torch
from torch.nn import  functional as F
from model.abstracRecommender import  GeneralRecommender
from model.MLP import MLPLayers

class MULTIDAE(GeneralRecommender):
    r"""MultiDAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

    We implement the the MultiDAE model with only user dataloader.
    """

    def __init__(self, config, dataGenerator):
        super(MULTIDAE, self).__init__(config, dataGenerator)

        self.layers = config.model_parameter.layers.value
        self.lat_dim = config.model_parameter.embeddingsize.value
        self.drop_out = config.model_parameter.dropout.value

        self.build_histroy_items(dataGenerator)

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [self.lat_dim] + self.encode_layer_dims[::-1][1:]

        self.encoder = MLPLayers(self.encode_layer_dims, activation="tanh")
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

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = self.encoder(h)
        return self.decoder(h)

    def calculate_loss(self, interaction):
        user = interaction

        rating_matrix = self.get_rating_matrix(user)

        z = self.forward(rating_matrix)

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        return ce_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]

        rating_matrix = self.get_rating_matrix(user)

        scores  = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def fullrank(self, interaction):
        user = interaction

        rating_matrix = self.get_rating_matrix(user)

        scores  = self.forward(rating_matrix)

        return scores