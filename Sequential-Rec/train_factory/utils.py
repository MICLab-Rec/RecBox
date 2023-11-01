import torch
import random
import numpy as np
from torch.utils.data import Sampler


def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]
    return torch.tensor(seq)


def gen_train_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    for e in src_seq:
        _, i_, _, _, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _, _, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size


def gen_eval_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    for e in src_seq:
        _, i_, _, _, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        data_size.append(min(max_len, len(_)))
    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _, _, _ = zip(*e)
        items.append(pad_sequence(i_, 1))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_size, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_size * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)