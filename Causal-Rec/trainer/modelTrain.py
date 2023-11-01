import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.getters import get_model,get_lossF
from trainer.Trainer import  Trainer
from utils.dataloader import GeneralData
from evaluator import evaluator,ranker
from torch.optim import  Adam
import seaborn
from utils.plots import plot_low_embedding,get_graph_from_adj,plot_hotmap,plot_line


def plot_causal_graph(trainer):
    # print(trainer.model.get_adj().cpu().detach().numpy())
    graph = trainer.model.get_adj(h_threshold= 0.5).cpu().detach().numpy()
    plot_hotmap(graph)
    get_graph_from_adj(graph)
    # plot_low_embedding(graph)


def train_model(config, datagenerator, model_per=None):
    dataGenerator = datagenerator
    Model,parameter = get_model(model_name= config.model_name)
    config.model_parameter = parameter
    model = Model(config,dataGenerator) if model_per is None else model_per
    # print(model)
    rank = ranker.Ranker(config,dataGenerator)
    stopper = evaluator.Evaluator(config)
    if config.explicit_loss:
        lossF = get_lossF(config.loss_name)
    else:
        lossF = None
    trainer = Trainer(model= model,ranker= rank,evaluator= stopper, optimizer= Adam,loss_func= lossF,dataGenerater= dataGenerator,config= config,DEVICE= config.Device)
    if not config.performance_rerun:
        trainer.fit(use_two_step= config.use_two_step)
    result = trainer.evaluate(save=config.save_result)
    if not config.performance_rerun:
        plot_line(x=range(len(trainer.loss_all)), y=trainer.loss_all, label='train loss')
        plot_line(x=range(len(trainer.validate_all)), y=trainer.validate_all, label='validate loss')
    if ((config.use_two_step or config.model_name == 'myvae') and config.hypertune) or config.plot_fig:
        plot_causal_graph(trainer)
    return result
