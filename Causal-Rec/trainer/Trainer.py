import numpy as np
from tqdm import  tqdm
from utils.tools import set_color,get_gpu_usage
import torch
import time

class Trainer(object):
    def __init__(self,model, ranker, evaluator, optimizer, loss_func,dataGenerater, config,DEVICE='cpu'):
        self.model = model.to(DEVICE)
        self.ranker = ranker
        self.evaluator = evaluator
        self.optimizer = optimizer(self.model.parameters(),lr = config.lr,weight_decay = config.weight_decay)
        self.lossF = loss_func
        self.lossname = config.loss_name.name if loss_func is not None else '-'
        self.trainloader = dataGenerater.trainDataloader()
        self.validateloader = dataGenerater.valDataloader()
        self.testloader = dataGenerater.testDataloader()
        self.epochs = config.epoch
        self.Device = DEVICE
        self.modelname = config.model_name
        self.data_name = config.data_name
        self.filename = '_'.join([config.data_name, config.model_name, self.lossname])
        # self.filename = '_'.join([config.data_name,config.model_name,config.loss_name.name,str(config.alpha)])
        self.validate_metric = config.validate_metric
        self.validate_k = config.validate_k
        self.validate_str = 0.0
        self.loss_result = 0.0
        self.save_result = config.save_result
        self.time = None
        self.loss_all = []
        self.validate_all =[]

    def train_one_epoch(self, epoch, start, early_stop=True):
        '''
        train model for one step
        @param epoch: current epoch
        @param early_stop: use early stop if True
        @param start: the train start time
        @return:
        '''

        self.model.train()
        loss_all = []
        with tqdm(
                self.trainloader,
                total=len(self.trainloader),
                ncols=80,
                bar_format='{desc}|{percentage:3.0f}%|{postfix}',
                desc=set_color(f"{self.modelname},{self.lossname},{self.data_name},{epoch}", 'green'),
                postfix=set_color(f'loss-{self.loss_result:.3f},val-{self.validate_str:.3f},gpu-{get_gpu_usage()},total time-{(time.time() - start) // 60} m {round((time.time() - start) % 60,0)} s', 'cyan'),
                leave= True,
            ) as traindata:
            # with torch.autograd.profiler.profile(enabled=True,use_cuda= True) as prof:
            for data in traindata:
                if data.device is not self.Device:
                    data = data.to(self.Device).long()

                if self.lossF is None:
                    loss = self.model.calculate_loss(data)
                else:
                    loss = self.lossF(self.model, data,self.Device)
                self.check_nan(loss)
                self.optimizer.zero_grad()
                loss.backward()
                loss_all.append(loss.detach().item())
                self.optimizer.step()
                self.loss_result = torch.mean(torch.Tensor(loss_all))
                # traindata.set_postfix()
            # cur_time = prof.key_averages().table(sort_by="self_cpu_time_total")

            # self.time = cur_time if self.time is None else self.time + cur_time
        if epoch > 10 and early_stop:
            validate = self.ranker.validate(self.model,self.testloader,self.validate_metric,self.validate_k)
            self.validate_str = validate
            self.validate_all.append(validate)
            self.loss_all.append(torch.mean(torch.Tensor(loss_all)))
            # traindata.set_postfix_str(set_color(f' validate result : {validate}', 'blue'))
            stop = self.evaluator.record_val(validate,epoch,self.model.state_dict())
            return stop
        else:
            return False

    def test_performance(self):
        self.ranker.evaluate(self.model,self.testloader)

    def first_stop_criterion(self, h_new, h_threshold, h_old):
        return h_new > h_threshold * h_old


    def fit(self, earlystop = True, use_two_step = False):
        stop = False
        epochs = 0
        start = time.time()
        h_old = np.inf
        epoch = 0
        if use_two_step:
            '''
            use the two step stopping criterion is use_two_step is True, the first step
            '''
            while epochs < 20:
                while self.model.get_rol() < 1e14:
                    print('*'*50, f'first step {epochs + 1} round', '*'*50)
                    for _ in range(20):
                        self.train_one_epoch(epoch=epoch, start=start, early_stop=True)
                        epoch += 1
                    h_new, h_threshold = self.model.get_h()
                    if self.first_stop_criterion(h_new, h_threshold, h_old):
                        self.model.update_rol()
                    else:
                        break
                    # stop = self.train_one_epoch(epoch=epoch, start=start, early_stop=True)
                print('-'*50)
                if h_new <= 1e-10 and epochs > 3:
                    break
                h_old = h_new.detach()
                self.model.update_alpha()
                epochs += 1
                self.model.global_state_change()
                for _ in range(20):
                    self.train_one_epoch(epoch=epoch, start=start, early_stop=True)
                    epoch += 1
                self.model.global_state_change()
        '''
        the second step
        '''
        if use_two_step:
            self.model.global_state_change()
        print('*' * 50, f'optimize for best performance', '*' * 50)
        for epoch in range(epoch, self.epochs + epoch):
            stop = self.train_one_epoch(epoch=epoch, early_stop=True, start=start)
            if stop and earlystop:
                self.evaluator.show_log()
                break
        self.evaluator.save_model(self.filename)
        if not stop:
            self.evaluator.show_log(earltStop=False)

        # self.test_performance()

    def fit_causal(self,earlystop = True):
        stop = False
        best_loss = np.Inf
        for epoch in range(self.epochs):
            stop = self.train_one_epoch(epoch)
            best_loss = min(self.loss_result,best_loss)
        self.evaluator.save_model(self.filename)
        if not stop:
            self.evaluator.show_log(earltStop= False)

        # self.test_performance()
        return best_loss

    def check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def evaluate(self,save = True):
        self.model.load_state_dict(self.evaluator.get_best_model(modelname= self.filename))
        self.ranker.evaluate(self.model,self.testloader)
        result = self.ranker.output_result()
        if self.save_result and save:
            self.ranker.save_result()
        return result
