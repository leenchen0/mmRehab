import torch
from torch import nn


def mean(list_tensors):
    return torch.mean(torch.stack(list(list_tensors)))


class JointsSimilarityLoss(nn.Module):
    def __init__(self, adaptive_weight=False):
        super(JointsSimilarityLoss, self).__init__()
        self.adaptive_weight = adaptive_weight

    def weighted_cos_sim(self, x, y):
        rho = torch.cosine_similarity(x - x.mean(), y - y.mean(), dim=-1)
        loss = 1 - rho
        return torch.mean(loss * self.std_weight(y))

    def std_weight(self, y):
        return torch.std(y, dim=1)

    def forward(self, out, labels):
        loss = mean([self.weighted_cos_sim(out[:, :, k, a], labels[:, :, k, a]) for k in range(out.shape[2]) for a in range(out.shape[3])])
        return loss

