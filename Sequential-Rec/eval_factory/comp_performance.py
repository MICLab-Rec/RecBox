import random
import numpy as np
from SeqRec.metric.topk_metrics import *


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rec_scores(data_loader, model, eval_data, args):
    hr_k = HitRatio(args.topk)
    ndcg_k = NDCG(args.topk)
    accuracy_k = Accuracy(args.topk)
    recall_k = Recall(args.topk)
    map_k = MAP(args.topk)
    model.eval()
    reset_random_seed(args.seed)
    with torch.no_grad():
        for _, (src_items, trg_items, data_size) in enumerate(data_loader):
            src = src_items.to(args.device)
            data_size = data_size.to(args.device)
            pred = model(src, data_size).to(device='cpu')
            target = trg_items
            index = torch.arange(target.size(0)).view(-1, 1)
            label = torch.zeros([target.size(0), eval_data.n_item]).bool()
            label = label.index_put((index, target), torch.ones_like(target).bool())
            hr_k(pred, label)
            ndcg_k(pred, label)
            accuracy_k(pred, label)
            recall_k(pred, label)
            map_k(pred, label)

    metrics = {'HR@{:d}'.format(args.topk): '{:.5f}'.format(hr_k.compute().item()),
               'NDCG@{:d}'.format(args.topk): '{:.5f}'.format(ndcg_k.compute().item()),
               'Accuracy@{:d}'.format(args.topk): '{:.5f}'.format(accuracy_k.compute().item()),
               'Recall@{:d}'.format(args.topk): '{:.5f}'.format(recall_k.compute().item()),
               'MAP@{:d}'.format(args.topk): '{:.5f}'.format(map_k.compute().item())}
    for key, value in metrics.items():
        print(key, '=', value)
    return metrics
