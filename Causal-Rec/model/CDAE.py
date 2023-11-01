r"""
CDAE
################################################
Reference:
    Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. In WSDM 2016.

Reference code:
    https://github.com/jasonyaw/CDAE
"""

import torch
import torch.nn as nn
from model.abstracRecommender import GeneralRecommender


class CDAE(GeneralRecommender):
    r"""Collaborative Denoising Auto-Encoder (CDAE) is a recommendation model
    for top-N recommendation that utilizes the idea of Denoising Auto-Encoders.
    We implement the the CDAE model with only user dataloader.
    """

    def __init__(self, config, dataset):
        super(CDAE, self).__init__(config, dataset)

        self.reg_weight_1 = config.model_parameter.reg_weight_1.value
        self.reg_weight_2 = config.model_parameter.reg_weight_2.value
        self.loss_type = config.model_parameter.loss_type.value
        self.hid_activation = config.model_parameter.hid_activation.value
        self.out_activation = config.model_parameter.out_activation.value
        self.embedding_size = config.model_parameter.embedding_size.value
        self.corruption_ratio = config.model_parameter.corruption_ratio.value

        self.build_histroy_items(dataset)

        if self.hid_activation == "sigmoid":
            self.h_act = nn.Sigmoid()
        elif self.hid_activation == "relu":
            self.h_act = nn.ReLU()
        elif self.hid_activation == "tanh":
            self.h_act = nn.Tanh()
        else:
            raise ValueError("Invalid hidden layer activation function")

        if self.out_activation == "sigmoid":
            self.o_act = nn.Sigmoid()
        elif self.out_activation == "relu":
            self.o_act = nn.ReLU()
        else:
            raise ValueError("Invalid output layer activation function")

        self.dropout = nn.Dropout(p=self.corruption_ratio)

        self.h_user = nn.Embedding(self.n_users, self.embedding_size)
        self.h_item = nn.Linear(self.n_items, self.embedding_size)
        self.out_layer = nn.Linear(self.embedding_size, self.n_items)

        # parameters initialization
        self.apply(self._xavier_uniform_initialization)

    def forward(self, x_items, x_users):
        h_i = self.dropout(x_items)
        h_i = self.h_item(h_i)
        h_u = self.h_user(x_users)
        h = torch.add(h_u, h_i)
        h = self.h_act(h)
        out = self.out_layer(h)
        return out

    def calculate_loss(self, interaction):
        x_users = interaction
        x_items = self.get_rating_matrix(x_users)
        predict = self.forward(x_items, x_users)

        if self.loss_type == "MSE":
            predict = self.o_act(predict)
            loss_func = nn.MSELoss(reduction="sum")
        elif self.loss_type == "BCE":
            loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError("Invalid loss_type, loss_type must in [MSE, BCE]")
        loss = loss_func(predict, x_items)
        # l1-regularization
        loss += self.reg_weight_1 * (
            self.h_user.weight.norm(p=1) + self.h_item.weight.norm(p=1)
        )
        # l2-regularization
        loss += self.reg_weight_2 * (
            self.h_user.weight.norm() + self.h_item.weight.norm()
        )

        return loss

    def predict(self, interaction):
        users = interaction[0]
        predict_items = interaction[1]

        items = self.get_rating_matrix(users)
        scores = self.forward(items, users)
        scores = self.o_act(scores)
        return scores[[torch.arange(len(predict_items)).to(self.device), predict_items]]

    def fullrank(self, interaction):
        users = interaction

        items = self.get_rating_matrix(users)
        predict = self.forward(items, users)
        predict = self.o_act(predict)
        return predict