from data_factory.data_process import InterData
from data_factory.utils import serialize, un_serialize
from train_factory.train_model import train
from model_config import SASRecConfig
from models.trs_based.sasrec import SASRec
from data_config import DataConfig
from train_config import TrainConfig
import numpy as np
import torch


data_args = DataConfig()
data_name = data_args.data_name
raw_data_pth = data_args.raw_data_path
clean_data_pth = data_args.clean_data_path

cold_item_threshold = data_args.cold_item_threshold
cold_user_threshold = data_args.cold_user_threshold
dataset = InterData(raw_data_pth, cold_item_threshold, cold_user_threshold)
count = 0
length = []
for seq in dataset.user_seq:
    count += len(seq)
    length.append(len(seq))

train_data, eval_data = dataset.partition(max_len=data_args.max_len)
print("#Users:", dataset.n_user - 1)
print("#Items:", dataset.n_item - 1)
print("#Interactions:", count)
print("Average Sequence Length:", np.mean(np.array(length)))
print("Sparsity:", 1 - count / ((dataset.n_user - 1) * (dataset.n_item - 1)))
serialize(dataset, clean_data_pth, False)

model_args = SASRecConfig()
model = SASRec(dataset.n_item, model_args)

train_args = TrainConfig()

assert data_args.max_len == model_args.max_len == train_args.max_len

model.to(train_args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=train_args.learning_rate)
train(model, optimizer, eval_data, train_data, train_args)
