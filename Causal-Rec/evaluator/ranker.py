import copy
import time

import pandas as pd
from openpyxl import  load_workbook
import  torch
from tqdm import tqdm

from utils.enmTypes import EvalType
import  numpy as np
import os

from utils.tools import set_color, get_gpu_usage


class Ranker(object):
    def __init__(self, config, dataGenerator):
        self.topk = config.topk
        self.val_topk = config.validate_k
        self.eval_type = config.eval_type
        self.val_type = config.val_type
        self.metrics = config.metrics
        self.save_path = config.path + f'/result/{config.data_name}.xlsx'
        self.dataname = config.data_name
        self.filename = f'{config.model_name}_{config.loss_name.name}'
        self.item_count = torch.FloatTensor(dataGenerator.get_pop())
        if config.use_price:
            self.price = torch.FloatTensor(dataGenerator.get_price())
        self.groundtruth_train = dataGenerator.getGroundtruth()
        self.tail_ratio = config.tail_ratio
        self.Device = config.Device


    def cal_preScore(self,model,batchdata,sorttype = EvalType.FULLSORT,validate = False):
        topk = max(self.topk) if not validate else self.val_topk
        model.eval()
        if sorttype == EvalType.FULLSORT:
            score = model.fullrank(batchdata)
            pos_matrix = torch.zeros_like(score, dtype=torch.int)
            for idx,user in enumerate(batchdata.cpu().numpy()):
                score[idx,self.groundtruth_train.loc[user]] = -np.inf
                pos_matrix[idx,self.groundtruth.loc[user]] = 1
            _, topk_idx = torch.topk(score,topk, dim = 1)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
        elif sorttype == EvalType.NEG_SAMPLE_SORT:
            score = model.predict(batchdata)
            _, topk_idx = torch.topk(score, topk)
            pos_matrix = torch.zeros_like(score, dtype=torch.int)
            pos_matrix[torch.zeros(1, dtype=torch.int).to(self.Device)] = 1
            pos_len_list = pos_matrix.sum()
            pos_idx = torch.gather(pos_matrix, dim=0, index=topk_idx)
        else:
            raise Exception('EvalType not exist, please check...........')


        return pos_idx.cpu().numpy(),pos_len_list.squeeze(-1).cpu().numpy(),topk_idx.cpu().numpy()

    def evaluate(self,model,testloader):
        model.eval()
        scores = []
        self.result = [{metric : [] for metric in self.metrics} for _ in range(len(self.topk))]
        if self.eval_type is EvalType.FULLSORT:
            self.groundtruth = testloader.dataset.groundtruth
        start = time.time()
        for batchdata in tqdm(
                testloader,
                total=len(testloader),
                ncols=80,
                bar_format='{desc}|{percentage:3.0f}%|{postfix}',
                desc=set_color(f"Test performance", 'green'),
                postfix=set_color(f'gpu-{get_gpu_usage()},total time-{(time.time() - start) // 60} m {round((time.time() - start) % 60,0)} s', 'cyan'),
                leave= True,
            ):
            if self.eval_type is EvalType.FULLSORT:
                batchdata = batchdata.to(self.Device).long()
            else:
                batchdata = [data.to(self.Device).long() for data in batchdata]
            pos_idx,pos_len_list,topk_idx = self.cal_preScore(model,batchdata,self.eval_type)
            scores.append([pos_idx, pos_len_list, topk_idx])
        for pos_idx, pos_len_list, topk_idx in scores:
            self.get_result(pos_idx,pos_len_list,topk_idx)

    def validate(self,model,dataloader,metric = 'recall', k = 30):
        model.eval()
        result = []
        scores = []
        start = time.time()
        if self.eval_type is EvalType.FULLSORT:
            self.groundtruth = dataloader.dataset.groundtruth
        for batchdata in tqdm(
                dataloader,
                total=len(dataloader),
                ncols=80,
                bar_format='{desc}|{percentage:3.0f}%|{postfix}',
                desc=set_color(f"Test performance", 'green'),
                postfix=set_color(f'gpu-{get_gpu_usage()},total time-{(time.time() - start) // 60} m {round((time.time() - start) % 60,0)} s', 'cyan'),
                leave= True,
            ):
            if self.eval_type is EvalType.FULLSORT:
                batchdata = batchdata.to(self.Device).long()
            else:
                batchdata = [data.to(self.Device).long() for data in batchdata]
            pos_idx,pos_len_list,topk_idx = self.cal_preScore(model,batchdata,self.val_type,validate= True)
            scores.append([pos_idx,pos_len_list,topk_idx])
        for pos_idx,pos_len_list,topk_idx in scores:
            res = eval('self.' + metric + '(pos_idx = pos_idx ,pos_len = pos_len_list)').mean(axis=0)
            result.append(res[k - 1])
        return round(np.mean(result),7)

    def  save_result(self, type='txt'):
        for id,k in enumerate(self.topk):
            for metric in self.metrics:
                res = format(np.mean(self.result[id][metric]),'.6f')
                self.result[id][metric] = res
        back = pd.DataFrame(self.result).T.rename(columns= {i : self.filename for i,k in enumerate(self.topk)}).T.reset_index()
        if not os.path.exists(self.save_path):
            pd.DataFrame(columns=self.metrics).to_excel(self.save_path, f'K = {self.topk[0]}')
            with pd.ExcelWriter(self.save_path, engine='openpyxl', mode='a') as writer:
                for i, k in enumerate(self.topk):
                    if i != 0:
                        pd.DataFrame(columns= self.metrics).to_excel(writer, f'K = {k}')
        # book = load_workbook(self.save_path)
        start = pd.read_excel(self.save_path,sheet_name=f'K = {k}',engine='openpyxl').shape[0]
        with pd.ExcelWriter(self.save_path,engine='openpyxl',mode= 'a',if_sheet_exists='overlay') as writer:
            # writer.book = book
            # writer.sheets.update( {sheet.title: sheet for sheet in book.worksheets})
            for i,k in enumerate(self.topk):
                back[i:i+1].to_excel(writer,f'K = {k}',startrow= start + 1,header= None,index= None)

    def get_result(self,pos_idx,pos_len_list,topk_idx):
        for metric in self.metrics:
            if metric in ['tail_percent','avg_pop','avg_price']:
                res = eval('self.' + metric + '(pos_idx = topk_idx ,pos_len = pos_len_list)').mean(axis=0)
            else:
                res = eval('self.' + metric + '(pos_idx = pos_idx ,pos_len = pos_len_list)').mean(axis=0)
            for id,k in enumerate(self.topk):
                self.result[id][metric].append(res[k - 1])


    def output_result(self, columns=None):#['hit', 'recall', 'ndcg']
        result = copy.deepcopy(self.result)
        for id,k in enumerate(self.topk):
            for metric in self.metrics:
                res = format(np.mean(self.result[id][metric]),'.6f')
                result[id][metric] = res
        back = pd.DataFrame(result).T.rename(columns= {i : f'topK={k}' for i,k in enumerate(self.topk)}).T
        if columns is None:
            print(back)
            return back
        else:
            print(back[columns])
            return back[columns]


    def hit(self,pos_idx, **args):
        r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
            you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
            we call it a hit.
                \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),
            :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
            :math:`\emptyset` denotes the empty set.
            """


        result = np.cumsum(pos_idx, axis=1)
        return (result > 0).astype(int)

    def mrr(self,pos_idx, **args):
        r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
    of the first relevant item found by an algorithm.
    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    .. math::
       \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}
    :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """


        idxs = pos_idx.argmax(axis=1)
        result = np.zeros_like(pos_idx, dtype=float)
        for row, idx in enumerate(idxs):
            if pos_idx[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


    def map(self, pos_idx, pos_len):
        r"""MAP_ (also known as Mean Average Precision) is meant to calculate
            average precision for the relevant items.
            Note:
                In this case the normalization factor used is :math:`\frac{1}{min(|\hat R(u)|, K)}`, which prevents your
                AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture
                all the correct ones.
            .. _MAP: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms
            .. math::
               \mathrm{MAP@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{min(|\hat R(u)|, K)} \sum_{j=1}^{|\hat{R}(u)|} I\left(\hat{R}_{j}(u) \in R(u)\right) \cdot  Precision@j)
            :math:`\hat{R}_{j}(u)` is the j-th item in the recommendation list of \hat R (u)).
            """

        pre = pos_idx.cumsum(axis=1) / np.arange(1, pos_idx.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_idx.astype(float), axis=1)
        len_rank = np.full_like(pos_len, pos_idx.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_idx, dtype=float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_idx.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


    def recall(self, pos_idx, pos_len):
        r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.
            .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall
            .. math::
               \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}
            :math:`|R(u)|` represents the item count of :math:`R(u)`.
            """


        return np.cumsum(pos_idx, axis= -1) / pos_len.reshape(-1, 1)


    def ndcg(self, pos_idx, pos_len):
        r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
           where positions are discounted logarithmically. It accounts for the position of the hit by assigning
           higher scores to hits at top ranks.
           .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
           .. math::
               \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
               \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})
           :math:`\delta(·)` is an indicator function.
           """


        len_rank = np.full_like(pos_len, pos_idx.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_idx, dtype=float)
        iranks[:, :] = np.arange(1, pos_idx.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_idx, dtype=float)
        ranks[:, :] = np.arange(1, pos_idx.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_idx, dcg, 0), axis=1)

        result = dcg / idcg
        return result


    def precision(self, pos_idx, **kwargs):
        r"""Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
            out of all the recommended items. We average the metric for each user :math:`u` get the final result.
            .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision
            .. math::
                \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}
            :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.
            """


        return pos_idx.cumsum(axis=1) / np.arange(1, pos_idx.shape[1] + 1)


    def avg_pop(self,pos_idx,**kwargs):
        r"""AveragePopularity computes the average popularity of recommended items.
            For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
            and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.
            .. math::
                \mathrm{AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}
            :math:`\phi(i)` is the number of interaction of item i in training data.
        """

        item_pop = np.zeros_like(pos_idx)
        for id, rows in enumerate(pos_idx):
            item_pop[id] = self.item_count[rows]
        return item_pop.cumsum(axis = 1) / np.arange(1, item_pop.shape[1] + 1)

    def avg_price(self,pos_idx, **kwargs):
        item_price = np.zeros_like(pos_idx)
        for id, rows in enumerate(pos_idx):
            item_price[id] = self.price[rows]
        print(item_price.cumsum(axis = 1)[0][49])
        return item_price.cumsum(axis = 1) / np.arange(1, item_price.shape[1] + 1)

    def tail_percent(self,pos_idx,**kwargs):
        if self.tail_ratio > 1:
            tail_items = self.item_count[self.item_count.count < self.tail_ratio].index.to_numpy()
        else:
            _,tail_items = torch.topk(self.item_count * -1,k= int(self.tail_ratio * self.item_count.shape[0]))
            # tail_items = self.item_count.index.to_numpy()[self.item_count['rank'].argsort().to_numpy()[:int(self.item_count.shape[0] * self.tail_ratio)]] #np.argsort(self.item_count)[: int(self.tail_ratio*self.item_count.shape[0])]
        value = np.zeros_like(pos_idx)
        for a,rows in enumerate(pos_idx):
            for b,item in enumerate(rows):
                if item in tail_items:
                    value[a][b] = 1
        return value.cumsum(axis=1) / np.arange(1, value.shape[1] + 1)

