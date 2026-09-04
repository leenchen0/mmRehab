import torch
from torch import nn


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, out, labels):
        x = out
        y = labels
        rho = torch.cosine_similarity(x - x.mean(), y - y.mean(), dim=-1).mean()
        loss = 1 - rho
        return loss

