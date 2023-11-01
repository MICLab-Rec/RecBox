import torch
from torchmetrics import Metric


class MRR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mrr", default=torch.zeros(1, dtype=torch.float))
        self.add_state("count", default=torch.zeros(1, dtype=torch.float))

    def update(self, pred, label):
        _, topk_indices = torch.topk(pred, pred.size(1), dim=1)
        reciprocal_rank = torch.zeros(label.size(0), dtype=torch.float)
        for i in range(label.size(0)):
            relevant_indices = (topk_indices[i] == label[i].nonzero().squeeze(1)).nonzero()
            if relevant_indices.numel() > 0:
                rank = relevant_indices[0, 0].item() + 1  # 获取第一个相关项目的排名
                reciprocal_rank[i] = 1.0 / rank

        reciprocal_rank[torch.isinf(reciprocal_rank)] = 0
        self.mrr += torch.sum(reciprocal_rank)
        self.count += label.size(0)

    def compute(self):
        average_mrr = self.mrr / self.count
        return average_mrr


class AUC(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("auc", default=torch.zeros(1, dtype=torch.float))
        self.add_state("count", default=torch.zeros(1, dtype=torch.float))
        self.positive_scores = []
        self.negative_scores = []

    def update(self, pred, label):
        positive_scores = pred[label == 1]
        negative_scores = pred[label == 0]
        #print(positive_scores)
        #print(negative_scores)

        # Calculate AUC for each batch
        batch_auc = self.calculate_auc(positive_scores, negative_scores)
        self.auc += batch_auc
        self.count += 1
        #print(self.auc)
        #print(self.count)

    def calculate_auc(self, positive_scores, negative_scores):
        # Calculate AUC for a single batch
        num_positives = len(positive_scores)
        num_negatives = len(negative_scores)

        auc = 0.0

        for pos_score in positive_scores:
            for neg_score in negative_scores:
                if pos_score > neg_score:
                    auc += 1
                elif pos_score == neg_score:
                    auc += 0.5

        #print(auc)
        auc /= (num_positives * num_negatives)

        return auc

    def compute(self):
        average_auc = self.auc / self.count
        return average_auc

