import torch.cuda

# from trainer.modelTrain import  train_model
from argParser import  parse_arg
from utils.dataloader import GeneralData
from utils.tools import setup_seed
from trainer.modelTrain import train_model
from trainer.heyperTune import HyperTune
import os
import numpy as np
from utils.enmTypes import LossType,InputType

if  __name__ == '__main__':
    config = parse_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config.Device =  'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'current device is {config.Device}...............')
    setup_seed(config.random_seed)
    config.data_name = 'tafeng'
    dataGeneral = GeneralData(config)
    config.model_name = 'myvae'
    train_model(config, dataGeneral)
