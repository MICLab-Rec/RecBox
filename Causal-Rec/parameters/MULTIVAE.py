from enum import Enum


class MULTIVAE(Enum):
    layers = [600]
    embeddingsize = 128
    dropout = 0.5
    anneal_cap = 0.2
    anneal_steps = 200000