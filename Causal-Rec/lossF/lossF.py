import torch
from torch import  nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from utils.getters import get_model
import os
import pickle

class LossF(nn.Module):
    def __init__(self):
        super(LossF, self).__init__()
        self.logsig = nn.LogSigmoid()

    def forward(self,data):
        pass

    def loss(self):
        pass



class MSE(LossF):
    def __init__(self,reduction = 'mean'):
        super(MSE, self).__init__()
        self.lossF = nn.MSELoss(reduction= reduction)

    def forward(self,model,data,device):
        prediction = model(data)
        return self.loss(prediction,data[2])

    def loss(self,pre,label):
        return self.lossF(pre,label)

class BCE(LossF):
    def __init__(self,reduction='none'):
        super(BCE, self).__init__()
        self.lossF = nn.BCELoss(reduction=reduction)

    def forward(self, model,data,device):
        prediction = model(data)
        return self.loss(prediction,data[2])

    def loss(self,pre,label):
        return self.lossF(pre,label)

    def cal_propensity(self):
        item_count = self.dataGenerator.item_count.reset_index()
        item_count.columns = ['item','count']
        max_count = item_count['count'].max()
        item_count['propensity_pos'] = item_count['count'].apply(lambda x: np.power(x / max_count,0.5))
        item_count['propensity_neg'] = item_count['count'].apply(lambda x: np.power(1 - x / max_count, 0.5))
        return item_count[['item','propensity_pos','propensity_neg']].set_index('item')


class BPR(LossF):
    def __init__(self,dataGenerator):
        super(BPR, self).__init__()
        self.dataGenerator = dataGenerator

    def forward(self,model,data,device):
        user = data[0].to(device)
        pos = data[1].to(device)
        neg = data[2].to(device)
        posScore = model([user,pos])
        negScore = model([user,neg])
        return self.loss(posScore,negScore).mean()


    def loss(self,posSc0re,negScore):
        return -self.logsig(posSc0re - negScore)

    def cal_propensity(self):
        item_count = self.dataGenerator.item_count.copy().reset_index()
        item_count.columns = ['item','count']
        max_count = item_count['count'].max()
        item_count['propensity_pos'] = item_count['count'].apply(lambda x: np.power(x / max_count,0.5))
        item_count['propensity_neg'] = item_count['count'].apply(lambda x: np.power(1 - x / max_count, 0.5))
        return item_count[['item','propensity_pos','propensity_neg']].set_index('item')


class UBPR(BPR):
    def __init__(self,dataGenerator):
        super(UBPR, self).__init__(dataGenerator)
        self.dataGenerator = dataGenerator
        self.propensity = self.cal_propensity()


    def forward(self,model,data,device):
        user = data[0].to(device)
        pos = data[1].to(device)
        neg = data[2].to(device)
        posScore = model([user,pos])
        negScore = model([user,neg])
        propsity_pos = torch.clamp(torch.FloatTensor(self.propensity.loc[pos.cpu().numpy()].propensity_pos.values),0.1,1.0).to(device)
        return torch.mul(1 / (propsity_pos + 1e-7),self.loss(posScore,negScore)).mean()


class RELMF(BCE):
    def __init__(self,dataGenerator):
        super(RELMF, self).__init__(dataGenerator)
        self.dataGenerator = dataGenerator
        self.lossF = nn.BCELoss(reduction= 'none')
        self.propensity = self.cal_propensity()

    def forward(self,model,data,device):
        data = [batch.to(device) for batch in data]
        score = model(data)
        weight = torch.FloatTensor(self.propensity.loc[data[1].cpu().numpy()].propensity_pos.values).to(device)
        return torch.mean(1 / (weight + 1e-7) * self.loss(score, data[2].to(torch.float)))


class EBPR(BPR):
    def __init__(self,dataGenerator):
        super(EBPR, self).__init__(dataGenerator)
        self.explainability = self.cal_explainability()

    def forward(self,model,data,device):
        user = data[0].to(device)
        pos = data[1].to(device)
        neg = data[2].to(device)
        posScore = model([user,pos])
        negScore = model([user,neg])
        explain_pos = torch.Tensor(self.explainability[user.cpu(),pos.cpu()]).to(device)
        explain_neg = torch.Tensor(self.explainability[user.cpu(), neg.cpu()]).to(device)

        return (self.loss(posScore,negScore) * explain_pos * (1 - explain_neg)).mean()

    def cal_explainability(self):
        path = os.getcwd() + '/data/' + self.dataGenerator.dataname + '/explainability.npy'
        if os.access(path,os.F_OK):
            return np.load(path)
        else:
            interaction_matrix = pd.crosstab(self.dataGenerator.train.user, self.dataGenerator.train.item)
            missing_columns = list(set(range(self.dataGenerator.datainfo.item_num)) - set(list(interaction_matrix)))
            missing_rows = list(set(range(self.dataGenerator.datainfo.user_num)) - set(interaction_matrix.index))
            for missing_column in missing_columns:
                interaction_matrix[missing_column] = [0] * len(interaction_matrix)
            for missing_row in missing_rows:
                interaction_matrix.loc[missing_row] = [0] * self.dataGenerator.datainfo.item_num
            interaction_matrix = np.array(interaction_matrix[list(range(self.dataGenerator.datainfo.item_num))].sort_index())
            item_similarity_matrix = cosine_similarity(interaction_matrix.T)
            np.fill_diagonal(item_similarity_matrix, 0)
            neighborhood = [np.argpartition(row, - 20)[- 20:] for row in item_similarity_matrix]
            explainability_matrix = np.array([[sum([interaction_matrix[user, neighbor] for neighbor in neighborhood[item]])
                                               for item in range(self.dataGenerator.datainfo.item_num)] for user in
                                              range(self.dataGenerator.datainfo.user_num)]) /20
            np.save(path,explainability_matrix,allow_pickle = True)
            return  explainability_matrix


class PDA(BPR):
    def __init__(self,dataGenerator):
        super(PDA, self).__init__(dataGenerator)
        self.propensity = self.cal_propensity()

    def forward(self,model,data,device):
        user = data[0].to(device)
        pos = data[1].to(device)
        neg = data[2].to(device)
        propsity_pos = torch.FloatTensor(self.propensity.loc[pos.cpu().numpy()].propensity_pos.values).to(device)
        propsity_neg = torch.FloatTensor(self.propensity.loc[neg.cpu().numpy()].propensity_pos.values).to(device)
        posScore = model([user, pos]) * propsity_pos
        negScore = model([user, neg]) * propsity_neg

        return self.loss(posScore, negScore).mean()


class UPL(BPR):
    def __init__(self,dataGenerator):
        super(UPL, self).__init__(dataGenerator)
        self.propensity = self.cal_propensity()
        self.load_Rel_MF()

    def forward(self,model,data,device):
        user = data[0].to(device)
        pos = data[1].to(device)
        neg = data[2].to(device)
        self.relMF.to(device)
        propsity_pos = torch.FloatTensor(self.propensity.loc[pos.cpu().numpy()].propensity_pos.values).to(device)
        propsity_neg = torch.FloatTensor(self.propensity.loc[neg.cpu().numpy()].propensity_neg.values).to(device)
        posScore = model([user, pos])
        negScore = model([user, neg])
        gama_neg = self.relMF([user,neg])

        return (self.loss(posScore,negScore) * (1 - gama_neg) * (1 / (propsity_pos * (1 - propsity_neg * gama_neg) + 1e-7))).mean()

    def load_Rel_MF(self):
        Model = get_model('MF')
        name = '_'.join([self.dataGenerator.config.data_name,'MF','RELMF'])
        self.relMF = Model(self.dataGenerator.config,self.dataGenerator.datainfo)
        self.relMF.load_state_dict(torch.load(self.dataGenerator.config.path + '/bestModel/' + name + '.pt'))


class MFDU(BCE):
    def __init__(self,dataGenerator):
        super(MFDU, self).__init__(dataGenerator)
        self.dataGenerator = dataGenerator
        self.lossF = nn.BCELoss(reduction= 'none')
        self.propensity = self.cal_propensity()

    def forward(self,model,data,device):
        propsity_pos = torch.clamp(torch.FloatTensor(self.propensity.loc[data[1].numpy()].propensity_pos.values),0.1,1.0).to(device)
        propsity_neg = torch.clamp(torch.FloatTensor(self.propensity.loc[data[1].numpy()].propensity_neg.values),0.1,1.0).to(device)
        data = [batch.to(device) for batch in data]
        score = model(data)
        weight = data[2] * propsity_pos + (1 - data[2]) * propsity_neg
        return (weight * self.loss(score,data[2].to(torch.float))).mean()


class DPR(BPR):
    def __init__(self,dataGenerate,alpha = 1):
        super(DPR, self).__init__(dataGenerate)
        self.alpha = alpha
        self.user_count = dataGenerate.datainfo.user_num
        self.cal_propensity()
        # self.propensity = self.cal_propensity()
        # self.propensity = (self.propensity - self.propensity.min()) / (self.propensity.max() - self.propensity.min())
        # self.propensity[self.propensity < 0.1] = 0.1


    def forward(self, model, data, device):
        user = data[0].to(device)
        pos = data[1].to(device)
        neg = data[2].to(device)
        propsity_pos = torch.FloatTensor(self.propensity.loc[pos.cpu().numpy()]['count'].values).to(device)
        propsity_neg = torch.FloatTensor(self.propensity.loc[neg.cpu().numpy()]['count'].values).to(device)
        posScore = model([user, pos]) / (propsity_pos)
        negScore = model([user, neg]) / (propsity_neg)

        return self.loss(posScore, negScore).mean()

    def cal_propensity(self):
        item_count = self.dataGenerator.item_count.copy().reset_index()
        item_count.columns = ['item', 'count']

        item_count['count'] = item_count['count'].apply(lambda x: (x / self.user_count))
        item_count['count'] = item_count['count'].apply(lambda x: np.power(x + 1,self.alpha))
        # item_count['count'] = item_count['count'].apply(lambda x: (1 / (x * self.alpha)))
        # all = item_count['count'].sum()
        # item_count['count'] = item_count['count'].apply(lambda x: x / all)
        item_count.set_index('item',inplace = True)
        self.propensity =  item_count


