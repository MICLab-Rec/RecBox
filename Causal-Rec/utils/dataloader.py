import logging
import os
import  numpy as np
import torch
from torch.utils.data import  Dataset,DataLoader
import  pandas as pd
from utils.enmTypes import  InputType,EvalType,DataStatic
from functools import  partial
from joblib import  Parallel,delayed
import scipy.sparse as sp
import copy


class DataPreprocess(object):
    def __init__(self,dataname,split_type = 'normal',percent = [0.7,0.1,0.2],compression = 'gzip',binary_threshold = 4,seed = 2023):
        '''
        @param dataname: the name of dataset
        @param split_type: arrays_like,defatlt is [train,test,validate] = [0.7,0.1,0.2]
        @param percent: the split formate, default is normal, other choose is [normal]
        @param compression: the zip type of data file
        @param seed: the random seed
        '''
        self.dataname = dataname
        self.split_type = split_type
        self.compression = compression
        self.percent = percent
        self.seed = seed
        self.path = os.getcwd() + '/data/'
        self.binary_threshold = binary_threshold

    def load_data(self):
        '''
        load and split the original interaction data
        @return:  trainSet,testSet,valSet
        '''
        if not os.path.exists(os.path.join(self.path,self.dataname)):
            raise FileNotFoundError('The Dataset is not exist. please check.......')
        if not os.path.exists(os.path.join(self.path,self.dataname,'train.pkl')):
            data = pd.read_pickle(self.path + self.dataname + '/data.pkl',compression= self.compression)
            static = {'name':self.dataname,'user': int(data.user.max()) + 1,'item': int(data.item.max()) + 1}
            pd.DataFrame([static]).to_pickle(self.path + self.dataname + '/static.pkl', compression=self.compression)
            train,test,val = self.train_test_split(data = data)
        else:
            train = pd.read_pickle(self.path + self.dataname + '/train.pkl', compression=self.compression)
            test = pd.read_pickle(self.path + self.dataname + '/test.pkl', compression=self.compression)
            val = pd.read_pickle(self.path + self.dataname + '/val.pkl', compression=self.compression)
        static = pd.read_pickle(self.path + self.dataname + '/static.pkl', compression=self.compression).to_dict(orient = 'list')

        self.binary_threshold = min(train.rating.max(),self.binary_threshold)
        info = DataStatic(static['name'],static['user'],static['item'])
        info.out()
        return  train,test,val,info

    def load_full_data(self):
        if not os.path.exists(os.path.join(self.path,self.dataname)) or not os.path.exists(os.path.join(self.path,self.dataname,'data.pkl')):
            raise FileNotFoundError('The Dataset is not exist. please check.......')

        data = pd.read_pickle(self.path + self.dataname + '/data.pkl',compression= self.compression)
        return data

    def load_price(self):
        if not os.path.exists(os.path.join(self.path, self.dataname, 'price.pkl')):
            raise FileNotFoundError('The Dataset is not exist or not contain price message. please check.......')
        else:
            price = pd.read_pickle(self.path + self.dataname + '/price.pkl', compression=self.compression)
            return price

    def train_test_split(self,data):
        '''
        give the split type and interaction data, spllit the data into particular type
        @param data:  inpult data
        @return: trainSet,testSet,valSet
        '''

        train,test,val = None,None,None
        if self.split_type == 'normal':
            grouped_inter_feat_index = self._grouped_index(data['user'].to_numpy())
            next_index = [[] for _ in range(len(self.percent))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=self.percent)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])

            train,val,test = [data.iloc[index] for index in next_index]
            train = pd.DataFrame(train,columns=['user','item','rating'])
            test = pd.DataFrame(test, columns=['user', 'item', 'rating'])
            val = pd.DataFrame(val, columns=['user', 'item', 'rating'])
            self.save([train,test,val],['train','test','val'])
        else:
            raise Exception('\033[1;35;40mSplit type is not exist , please choose the exist one...............\033[0m')
        print(f'\033[0;35;40m After split, the trainset size is {train.shape},valset size is {val.shape}, testset size is {test.shape}\033[0m')
        return  train,test,val

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index.values()

    def _calcu_split_ids(self, tot, ratios):
        '''
        Given split ratios, and total number, calculate the number of each part after splitting.
        Other than the first one, each part is rounded down.
        @param tot: Total number.
        @param ratios:  List of split ratios. No need to be normalized.
        @return: Number of each part after splitting.
        '''
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        for i in range(1, len(ratios)):
            if cnt[0] <= 1:
                break
            if 0 < ratios[-i] * tot < 1:
                cnt[-i] += 1
                cnt[0] -= 1
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def copy(self, new_inter_feat):
        '''
        Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.
        @param new_inter_feat: The new interaction feature need to be updated.
        @return: the new :class:`~Dataset` object, whose interaction feature has been updated.
        '''
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def save(self,data :list, name:list):
        if not isinstance(data[0],pd.DataFrame):
            raise TypeError('\033[1;35;40m The input data must be the list of dataframe..............\033[0m')
        if len(data) != len(name):
            raise ValueError('\033[1;35;40m The size of name and data must match.......\033[0m')
        for idx,df in enumerate(data):
            df.to_pickle(self.path + self.dataname + f'/{name[idx]}.pkl', compression=self.compression)
        print(f'\033[0;35;40m Total save {len(name)} files, contains {name}\033[0m')

    def load_negative(self):
        if os.access(self.path + self.dataname + '/negative.pkl',mode= os.F_OK):
            negative = pd.read_pickle(self.path + self.dataname + '/negative.pkl', compression=self.compression)
            return negative
        else:
            print('no exist negative file, begain to genreate.......')
            raise FileExistsError('no exist negative file, begain to genreate.......')







class GeneralData(object):
    def __init__(self, config,validate_percent = 0.3):
        self.config = config
        self.dataname = config.data_name
        self.dataPreprocess = DataPreprocess(dataname= config.data_name,split_type= config.split_type, percent= config.split_percent,
                                             compression= config.compression,binary_threshold= config.binary_threshold,seed = config.random_seed)
        self.train, self.test, self.val,self.datainfo = self.dataPreprocess.load_data()
        self.data_type = config.data_type
        self.evalType = config.eval_type
        self.valType = config.val_type
        self.batch_size = config.batchsize
        self.num_worker = config.num_worker
        self.negative_generate = config.negative_generate
        self.negative_number = config.negative_number
        self.negative_sample_sort_nums = config.negative_sample_sort_nums
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        np.random.seed(self.config.random_seed)



    def sample(self,dataSet,negative_num = None, concat = False):
        '''
        sample negatives for pairwise trainset and sample-based evaluation
        @param dataSet: the input dataset, train or test set or var set
        @param concat: if true, the negative samples will concat with orignal one, and sort by rating and postives items id as flag to guarantee the first is postive items
        @param negative_num: the negative number of each user-item pair, if nane, the default will as config set
        @return: the set after negative sample,
        '''
        try:
            if self.negative_generate:
                print('Begain to genreate new negative samples as request..........')
                raise Exception('Begain to genreate new negative samples as request..........')
            data =self.dataPreprocess.load_negative()
        except Exception:
            path = os.getcwd() + '/data/'
            data = self.dataPreprocess.load_full_data()
            item_pool = set(data.item.unique())
            interact_status = data.groupby('user')['item'].apply(set).reset_index().rename(columns={'item': 'postive_items'})
            interact_status['negative_items'] = interact_status['postive_items'].apply(lambda x: item_pool - x)
            data = interact_status[['user', 'postive_items','negative_items']].sort_values(by = 'user', ascending= True,inplace = True)
            data.to_pickle(path + self.dataname + '/negative.pkl', compression='gzip')
        if negative_num is not None:
            times = negative_num
        else:
            times = self.negative_number
        backdata = pd.DataFrame(np.repeat(dataSet.values,repeats= times,axis= 0),columns=['user','item','rating'])[['user','item']]
        negatives =np.array( data.loc[backdata.user.tolist()].negative_items.apply(lambda x: np.random.choice(list(x), times ,replace = False).tolist()).tolist()).flatten()
        backdata['negative'] = pd.Series(negatives)
        if concat:
            dataSet = dataSet.copy()
            dataSet['flag'] = dataSet.item
            dataSet = dataSet[['user','item','flag','rating']]
            backdata = backdata[['user','negative','item']]
            backdata.columns = ['user', 'item', 'flag']
            backdata['rating'] = np.zeros(backdata.shape[0])
            backdata.columns = ['user','item','flag','rating']
            backdata = pd.concat([dataSet,backdata])
            backdata.sort_values(by = ['user','flag','rating'],inplace= True, ascending=[True,True,False])
            backdata.reset_index(inplace= True, drop= True)
        return backdata[['user','item']]

    def get_pop(self):
        item_count = self.train.groupby('item')['user'].count()
        item_count.colums = pd.Series(['count'])
        pop = np.zeros(self.datainfo.item_num)
        pop[item_count.index] = item_count.values
        return pop

    def get_price(self):
        price = self.dataPreprocess.load_price()
        item_price = price.groupby('item')['price'].mean()
        # print(item_price)
        item_price.colums = pd.DataFrame(item_price)
        # print(item_price.head())
        price = np.zeros(self.datainfo.item_num)
        price[item_price.index] = item_price.values
        print(min(price),max(price))
        return price

    def histiory_items(self,value_field = None,max_history_len = None,row = 'user'):
        '''
        get the history items for each user
        @return: coo matrix,row: user , col: item
        '''
        dataset = self.binary(name= 'history need')
        inter_feat = dataset.loc[dataset.rating > 0]
        # inter_feat.shuffle()
        user_ids, item_ids = (
            inter_feat['user'].to_numpy(),
            inter_feat['item'].to_numpy(),
        )
        if value_field is None:
            values = np.ones(len(inter_feat))
        else:
            if value_field not in inter_feat:
                raise ValueError(
                    f"Value_field [{value_field}] should be one of `inter_feat`'s features."
                )
            values = inter_feat[value_field].numpy()

        if row == "user":
            row_num, max_col_num = self.datainfo.user_num, self.datainfo.item_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.datainfo.item_num, self.datainfo.user_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        max_inter_num = np.max(history_len)
        if max_history_len is not None:
            col_num = min(max_history_len, max_inter_num)
        else:
            col_num = max_inter_num

        # if col_num > max_col_num * 0.2:
        #     self.logger.warning(
        #         f"Max value of {row}'s history interaction records has reached "
        #         f"{col_num / max_col_num * 100}% of the total."
        #     )

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            if history_len[row_id] >= col_num:
                continue
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return (
            torch.LongTensor(history_matrix),
            torch.FloatTensor(history_value),
            torch.LongTensor(history_len),
        )
        # nonzero = dataset.groupby('user')['item'].apply(set).sort_values()
        # nonzero = nonzero.apply(list)
        # nonzero.columns = ['item']
        # nonzero = nonzero.apply(list)
        # userdict = {u:[] for u in range(self.datainfo.user_num)}
        # for u in nonzero.index:
        #     userdict[u] = nonzero.loc[u]
        # print(np.array(itemgetter(*[0,1])(userdict)))
        # usermatrix = sp.coo_matrix(
        #     (np.ones(dataset.shape[0]),(dataset.user.to_numpy(),dataset.item.to_numpy())),
        #     shape=(self.datainfo.user_num,self.datainfo.item_num),dtype = np.int16
        # ).tocsr()
        # # print(usermatrix[[0,1]].todense())
        # return usermatrix

    def getGroundtruth(self):
        data = self.train.groupby('user')['item'].apply(set).reset_index().rename(columns={'item': 'postive_items'})
        data.sort_values(by = 'user', ascending= True,inplace= True)
        data = data['postive_items'].apply(list)
        return data

    def trainDataloader(self):
        if self.config.data_binary or self.config.data_type is InputType.PAIRWISE:
            #Pair wise data needs binary data
            trainSet = self.binary()
        if self.config.data_type is InputType.POINTWISE:
            dataset = GeneralDataset(trainSet)
        elif self.config.data_type is InputType.PAIRWISE:
            trainSet = trainSet[trainSet.rating > 0]
            trainSet = self.sample(trainSet)
            dataset = PairwiseDataset(trainSet)
        elif self.config.data_type is InputType.USERWISE:
            dataset = UserwiseDataset(trainSet)
        else:
            raise TypeError(f'\033[0;35;40m InputType is not regulated........\033[0m')
        return DataLoader(dataset= dataset, num_workers= self.num_worker, pin_memory= False,batch_size= self.batch_size,shuffle= True,prefetch_factor= 4 * self.num_worker)

    def testDataloader(self):
        testSet = self.binary(self.test.copy(),name='testset')
        testSet = testSet[testSet.rating > 0]
        if self.evalType == EvalType.FULLSORT:
            dataset = UserwiseDataset(testSet)
            return DataLoader(dataset=dataset, num_workers = self.num_worker, pin_memory=False, batch_size=self.batch_size, shuffle=True,prefetch_factor= 4 * self.num_worker)
        elif self.evalType == EvalType.NEG_SAMPLE_SORT:
            testSet = self.sample(testSet,negative_num= self.negative_sample_sort_nums,concat= True)
            dataset = NegSampleDataset(testSet)
            self.batch_size = self.negative_sample_sort_nums + 1
            return DataLoader(dataset=dataset, num_workers = self.num_worker, pin_memory=False, batch_size=self.batch_size, shuffle=False,prefetch_factor= 4 * self.num_worker)
        else:
            raise Exception('data type is render, please check......... ')

    def valDataloader(self):
        valSet = self.binary(self.val.copy(),name= 'valset')
        valSet = valSet[valSet.rating > 0]
        if self.valType == EvalType.FULLSORT:
            dataset = UserwiseDataset(valSet)
            return DataLoader(dataset=dataset, num_workers = self.num_worker, pin_memory=False, batch_size=self.batch_size, shuffle=True,prefetch_factor= 4 * self.num_worker)
        elif self.valType == EvalType.NEG_SAMPLE_SORT:
            valSet = self.sample(valSet,negative_num= self.negative_sample_sort_nums,concat= True)
            dataset = NegSampleDataset(valSet)
            self.batch_size = self.negative_sample_sort_nums + 1
            return DataLoader(dataset=dataset, num_workers = self.num_worker, pin_memory=False, batch_size=self.batch_size, shuffle=False,prefetch_factor= 4 * self.num_worker)
        else:
            raise Exception('data type is render, please check......... ')

    def createSparseGraph(self):
        path = os.getcwd() + '/data/' + self.dataname + '/sparse_graph.pt'
        if os.access(path, os.F_OK):
            return torch.load(path)
        else:
            A = sp.dok_matrix((self.datainfo.user_num + self.datainfo.item_num, self.datainfo.user_num + self.datainfo.item_num), dtype=np.float32)
            UserItemNet = sp.coo_matrix((np.ones(len(self.train['user'])), (self.train['user'], self.train['item'])),
                                     shape=(self.datainfo.user_num, self.datainfo.item_num))
            UserItemNet_t = UserItemNet.transpose()
            data_dict = dict(
                zip(zip(self.train.user, self.train.item + self.datainfo.user_num), [1] * self.train.shape[0])
            )
            data_dict.update(
                dict(
                    zip(
                        zip(self.train.user + self.datainfo.user_num, self.train.item),
                        [1] * self.train.shape[1],
                    )
                )
            )
            A._update(data_dict)
            # norm adj matrix
            sumArr = (A > 0).sum(axis=1)
            # add epsilon to avoid divide by zero Warning
            diag = np.array(sumArr.flatten())[0] + 1e-7
            diag = np.power(diag, -0.5)
            D = sp.diags(diag)
            L = D * A * D
            # covert norm_adj matrix to tensor
            L = sp.coo_matrix(L)
            row = L.row
            col = L.col
            i = torch.LongTensor(np.array([row, col]))
            data = torch.FloatTensor(L.data)
            SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
            torch.save(SparseL,path)
            return SparseL

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def binary(self,dataset = None,name = None):
        '''

        @param dataset: the dataset need to binarry,default is train set
        @return:
        '''
        if dataset is not None:
            trainSet = dataset
        else:
            trainSet = self.train.copy()
            name = 'trainSet' if name is None else name
        trainSet.rating.loc[trainSet.rating < self.dataPreprocess.binary_threshold] = 0
        trainSet.rating.loc[trainSet.rating >= self.dataPreprocess.binary_threshold] = 1
        print(f'\033[0;35;40m {name} is binaried by threshold value {self.dataPreprocess.binary_threshold}\033[0m')
        return trainSet





class GeneralDataset(Dataset):
    def __init__(self,data):
        self.data = data.to_numpy()

    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        '''
        @param id:
        @return:  user,item,label
        '''
        return self.data[id][0].astype(np.int32),self.data[id][1].astype(np.int32),self.data[id][2].astype(np.int32)

class PairwiseDataset(Dataset):
    def __init__(self,data):
        self.data = data.to_numpy()


    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        '''
        @param id:
        @return:  user,postive,negative
        '''
        return self.data[id][0].astype(np.int32),self.data[id][1].astype(np.int32),self.data[id][2].astype(np.int32)

class UserwiseDataset(Dataset):
    def __init__(self,data):
        self.data = data.user.unique()
        self.groundtruth = self._getGroundtruth(data)


    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        '''
        @param id:
        @return:  user
        '''
        return self.data[id].astype(np.int32)
    def _getGroundtruth(self,data):
        groundtruth = data.groupby('user')['item'].apply(set).reset_index().rename(columns={'item': 'postive_items'})
        groundtruth.sort_values(by = 'user', ascending= True,inplace= True)
        groundtruth['postive_items'] = groundtruth['postive_items'].apply(list)
        groundtruth.set_index('user',inplace= True)
        return  groundtruth


class FullSortDataset(Dataset):
    def __init__(self,data,info):
        self.info = info
        self.data,self.groundTruth,self.mask,self.max_mask_Len, self.max_pos_Len = self.generate(data)

    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        return self.data[id],list(self.mask[self.data[id]]),list(self.groundTruth[self.data[id]])

class NegSampleDataset(Dataset):
    def __init__(self,data):
        self.data = data.to_numpy()

    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        return self.data[id][0].astype(np.int32),self.data[id][1].astype(np.int32)



