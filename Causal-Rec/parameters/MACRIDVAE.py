from enum import Enum


class MACRIDVAE(Enum):
    encoder_hidden_size = [600]
    embeddingsize = 128
    dropout = 0.5
    kfac = 10
    nogb = False
    std = 0.01
    tau = 0.1
    anneal_cap = 0.2
    anneal_steps = 200000
    reg_weights = [0.0,0.0]