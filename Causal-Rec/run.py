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
    config.hypertune = False
    setup_seed(config.random_seed)
    config.data_name = 'tafeng'
    dataGeneral = GeneralData(config)
    config.model_name = 'myvae'
    config.save_result = False
    config.performance_rerun = True
    config.use_two_step = False
    config.plot_fig = False
    config.metrics = ['ndcg']
    config.use_price = False
    train_model(config, dataGeneral)
    # for data in ['ml-100k','ml-1m','ml-10m','reasoner','tafeng','epinions','lastfm','yelp','ml-20m']:#'ml-100k','ml-10m','ml-1m','reasoner','tafeng','epinions','lastfm','yelp','ml-20m','ifashion'
    #     config.data_name = data
    #     dataGeneral = GeneralData(config)
    #     config.performance_rerun = False
    #     for model in ['myvae']:
    #         setup_seed(config.random_seed)
    #         config.model_name = model
    #         config.file_name = '_'.join([model, data])
    #         train_model(config, dataGeneral)
    # for data in ['ml-20m']:#'ml-100k','ml-10m','ml-1m','reasoner','tafeng','epinions','lastfm','yelp','ml-20m','ifashion'
    #     config.data_name = data
    #     dataGeneral = GeneralData(config)
    #     config.performance_rerun = True
    #     config.save_result = True
    #     for model in ['multivae','multidae','macridvae','recvae','CDAE','betavae','betatcvae']:#['multivae','multidae','macridvae','recvae','CDAE','betavae','betatcvae']
    #         setup_seed(config.random_seed)
    #         config.model_name = model
    #         config.file_name = '_'.join([model, data])
    #         train_model(config, dataGeneral)

    if config.hypertune:
        hypertune = HyperTune(hypername=config.hypername, hypervalue=config.hypervalue, modelname='myvae', datasets=config.hyperdatasets)
        hypertune.tune()
