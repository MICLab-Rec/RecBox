import torch.nn.functional as F


class TrainConfig(object):
    def __init__(self, **kwargs):
        self.seed = kwargs.pop('seed', 42)
        self.num_workers = kwargs.pop('num_workers', 12)
        self.prefetch_factor = kwargs.pop('prefetch_factor', 2)
        self.train_batch_size = kwargs.pop('train_batch_size', 512)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 256)
        self.max_len = kwargs.pop('max_len', 64)
        self.result_path = kwargs.pop('result_path',
                                      '/home/admin/桌面/MICRecBox/SeqRec/overall_recommendation_performance/lsbn/nyc/SASRec.txt')
        self.model_path = kwargs.pop('model_path',
                                     '/home/admin/桌面/MICRecBox/SeqRec/trained_sequential_recommenders/lbsn/nyc/SASRec.pkl')
        self.device = kwargs.pop('device', 'cuda:0')
        self.loss_fn = kwargs.pop('loss_fn', F.cross_entropy)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.topk = kwargs.pop('topk', 10)
        self.num_epoch = kwargs.pop('num_epoch', 100)
