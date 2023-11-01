import csv
import os
from enum import Enum

import torch

from argParser import parse_arg
from parameters.MYVAE import MYVAE
from utils.dataloader import GeneralData
from utils.getters import get_model
from utils.tools import setup_seed
from itertools import product
from trainer.modelTrain import train_model

class HyperTune(object):
    def __init__(self, hypername, hypervalue, datasets, modelname):
        '''
        @param hypername: the hyper parameters to tune, list
        @param hypervalue: the hyper parameters value range, list
        @param datasets: the datasets used, list
        @param modelname: the name of the model
        '''

        self.hyper_name = hypername
        self.hyper_value = list(product(*hypervalue))
        self.datasets = datasets
        self.modelname = modelname
        self.config = parse_arg()
        self.config.model_name = modelname
        self.filename = ''

    def tune(self):
        self.config.Device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for dataset in self.datasets:
            self.filename =os.getcwd() +'/hyper_result/hypertune-'+ self.modelname + '-' + dataset + '.txt'
            print(self.filename)
            for value in self.hyper_value:
                setup_seed(self.config.random_seed)
                self.config.data_name = dataset
                dataGeneral = GeneralData(self.config)
                self.config.save_result = False
                Model, parameter = get_model(model_name=self.modelname)
                text = ''
                parameter_dict = {}
                for k, v in parameter.__members__.items():
                    parameter_dict[k] = v.value
                for k, name in enumerate(self.hyper_name):
                    parameter_dict[name] = value[k]
                    text += name+': '+str(value[k])+'\n'
                print('*'*50)
                print(text)
                print('*'*50)
                parameter = Enum(self.modelname.upper(), parameter_dict)
                self.config.model_parameter = parameter
                model = Model(self.config, dataGeneral)
                result = train_model(self.config, dataGeneral, model_per=model)
                self.save(text=text, res=result)

    def save(self, res=None, text=None):
        if text is not None:
            with open(self.filename, mode='a', encoding='utf-8') as file:
                file.write(text)
        if res is not None:
            res.to_csv(self.filename, sep=' ', mode='a', header=False, quoting=csv.QUOTE_NONE, escapechar=' ')

