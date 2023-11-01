import  numpy as np
import  copy
import torch
import os
from matplotlib import  pyplot as plt

class Evaluator:
    def __init__(self, config ):
        self.sortType = config.sortType
        self.total_epoch = config.epoch
        self.val_loss = []
        self.epochs = []

        self.val_best = np.inf if self.sortType == "ascending" else -np.inf

        self.best_val_model = None
        self.best_val_epoch = 0

        self.patience_max = config.patience_max
        self.patience_counter = 0
        self.reach_best_epoch = None
        self.modelname = config.model_name

        self.path = config.path + '/bestModel/'


    def record_val(self, performance, epoch, state_dict):
        self.patience_counter += 1
        self.val_loss.append(performance)

        if self.sortType == "ascending":
            if performance < self.val_best - 1e-7:
                self.val_best = performance
                self.best_val_model = copy.deepcopy(state_dict)
                self.patience_counter = 0
                self.reach_best_epoch = epoch

        elif self.sortType == "descending":
            if performance > self.val_best + 1e-7:
                self.val_best = performance
                self.best_val_model = copy.deepcopy(state_dict)
                self.patience_counter = 0
                self.reach_best_epoch = epoch
        else:
            raise Exception("invalid sort type, please check...............")

        if self.patience_counter >= self.patience_max:
            return True
        return False

    def get_best_model(self,modelname = None):
        return self.best_val_model if self.best_val_model is not None else self.load_model_state(modelname)

    def load_model_state(self,modelname):
        return torch.load(self.path+ modelname + '.pt')

    def show_log(self,earltStop = True):
        if earltStop:
            print('reach max patience {}  at {}, best validate performance is {}'.format(self.patience_max,self.reach_best_epoch,self.val_best))
        else:
            print('run out of all epochs {} without earlystop, best validate performance is {}'.format(self.total_epoch, self.val_best))

    def save_model(self,filename):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.best_val_model,self.path+ filename + '.pt')

    def plot(self,modelname,epoch,savefig= False):
        if epoch != None:
            self.epochs.append(epoch)
        plt.clf()
        plt.subplot(211)
        plt.plot(range(len(self.val_loss)),self.val_loss,lw = '2')
        plt.title("val recall--" + modelname)
        plt.title("train Loss")
        plt.tight_layout()
        plt.pause(0.1)
        if savefig:
            plt.savefig(os.getcwd()+'/evaluate/log_png/' +modelname +'.pdf', bbox_inches='tight')

    def reset_patience(self):
        self.patience_counter = 0
