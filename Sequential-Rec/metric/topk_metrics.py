import torch
from torchmetrics import Metric


class HitRatio(Metric):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.add_state("hits", default=torch.zeros(1, dtype=torch.float))
        self.add_state("total", default=torch.zeros(1, dtype=torch.float))

    def update(self, pred, label):
        topk_pred = torch.topk(pred, self.topk, dim=1)[1]
        batch_hits = torch.sum(label[torch.arange(label.size(0)).unsqueeze(1), topk_pred], dim=1)
        self.hits += torch.sum(batch_hits)
        self.total += label.sum()

    def compute(self):
        hit_ratio = self.hits / self.total
        return hit_ratio


class NDCG(Metric):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.add_state("ndcg", default=torch.zeros(1, dtype=torch.float))
        self.add_state("count", default=torch.zeros(1, dtype=torch.float))

    def update(self, pred, label):
        _, topk_indices = torch.topk(pred, self.topk, dim=1)
        topk_labels = label[torch.arange(label.size(0)).unsqueeze(1), topk_indices]
        dcg = (1 / torch.log2(torch.arange(2, topk_labels.size(1) + 2).double()))*topk_labels
        dcg = torch.sum(dcg, dim=1, keepdim=True)
        idcg_list = []
        for i in range(label.size(0)):
            num_relevant_items = label[i].sum().item()
            idcg = torch.sum(1 / torch.log2(torch.arange(2, num_relevant_items + 2).double()), dim=0)
            idcg_list.append(idcg)

        idcg_tensor = torch.stack(idcg_list).view(-1, 1)
        ndcg = dcg / idcg_tensor
        self.ndcg += torch.sum(ndcg)
        self.count += label.size(0)

    def compute(self):
        average_ndcg = self.ndcg / self.count
        return average_ndcg


class Recall(Metric):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.add_state("recall", default=torch.zeros(1, dtype=torch.float))
        self.add_state("count", default=torch.zeros(1, dtype=torch.float))

    def update(self, pred, label):
        # Calculate Recall for each user in the batch
        _, topk_indices = torch.topk(pred, self.topk, dim=1)
        batch_hits = torch.sum(label[torch.arange(label.size(0)).unsqueeze(1), topk_indices], dim=1)
        batch_recall = batch_hits / label.sum(dim=1)
        self.recall += torch.sum(batch_recall)
        self.count += label.size(0)

    def compute(self):
        # Calculate the average Recall
        average_recall = self.recall / self.count
        return average_recall


class Accuracy(Metric):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.add_state("accuracy", default=torch.zeros(1, dtype=torch.float))
        self.add_state("count", default=torch.zeros(1, dtype=torch.float))

    def update(self, pred, label):
        # Calculate Accuracy for each user in the batch
        _, topk_indices = torch.topk(pred, self.topk, dim=1)
        batch_correct = torch.sum(label[torch.arange(label.size(0)).unsqueeze(1), topk_indices], dim=1)
        batch_accuracy = batch_correct / self.topk
        self.accuracy += torch.sum(batch_accuracy)
        self.count += label.size(0)

    def compute(self):
        # Calculate the average Accuracy
        average_accuracy = self.accuracy / self.count
        return average_accuracy


class MAP(Metric):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.add_state("average_precision", default=torch.zeros(1, dtype=torch.float))
        self.add_state("count", default=torch.zeros(1, dtype=torch.float))

    def update(self, pred, label):
        _, topk_indices = torch.topk(pred, self.topk, dim=1)

        batch_average_precision = 0.0
        for i in range(label.size(0)):
            correct_predictions = torch.sum(label[i, topk_indices[i]])
            #print(torch.cumsum(label[i, topk_indices[i]].float(), dim=0))
            #print(torch.arange(self.topk) + 1)
            if correct_predictions == 0:
                average_precision = 0.0
            else:
                precision_at_k = torch.cumsum(label[i, topk_indices[i]].float(), dim=0) / (torch.arange(self.topk) + 1).float()
                #print(precision_at_k)
                correct_indices = torch.nonzero(label[i, topk_indices[i]]).squeeze()
                average_precision = (correct_predictions / self.topk) * torch.sum(precision_at_k[correct_indices])

            batch_average_precision += average_precision

        self.average_precision += batch_average_precision
        self.count += label.size(0)

    def compute(self):
        mean_average_precision = self.average_precision / self.count
        return mean_average_precision