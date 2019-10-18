import torch
# from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F


class TopKAccuracy(nn.Module):
    """
    """

    def __init__(self, k=1):
        self.k = k
        super(TopKAccuracy, self).__init__()

    def forward(self, input, target):
        gt_scores = input.gather(1, target.view(-1, 1))
        top_k_scores, _ = torch.topk(input, self.k, dim=1)
        top_k_threshold, _ = torch.min(top_k_scores, dim=1)
        return torch.mean((gt_scores >= top_k_threshold).float())
