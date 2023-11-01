from enum import Enum


class MYVAE(Enum):
    encoder_hidden_size = [600,400,256]
    embeddingsize = 128
    concepts = 4
    concepts_k = 4
    concepts_dim = 128
    dropout = 0.5
    std = 0.01
    tau = 0.2
    anneal_cap = 0.2
    anneal_steps = 200000
    reg_weights = [0.0,0.0]
    lambda_A = 0.
    alpha = 50
    rol = 1e-2
    rol_mutiply = 10
    h_threshold = 0.25
    c_A = 50.
    beta = 10
    l1_a = 10
    seed = 1228
    mask_all = True
    mask_local = False
    mask_c = None

    @classmethod
    def change(cls, name, value):
        setattr(cls, name, value)