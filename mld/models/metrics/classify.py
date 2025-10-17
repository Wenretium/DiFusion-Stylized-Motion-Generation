from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric, Accuracy
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class ClassifyMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 topk=1,
                 **kwargs):
        super().__init__()

        self.name = "ClassifyMetrics"
        self.metrics = []

        self.metrics.append("Accuracy")
        
        self.add_state("Accuracy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        self.topk = topk
        print(f"Compute top{topk} style classification accuracy")

    def compute(self, sanity_flag):
        
        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        
        # if in sanity check stage then jump
        if sanity_flag:
            return metrics
        
        metrics["Accuracy"] = self.correct.float() / self.total

        return {**metrics}

    def update(
        self,
        pred_label,
        gt_label,
    ):
        # Top 1
        # pred_label = pred_label.argmax(1)
        # self.correct += torch.sum(pred_label == gt_label)
        # Top k
        _, pred_label = pred_label.topk(self.topk, 1, True, True)
        gt_label = gt_label.view(-1, 1)
        # print(pred, gt_label)
        # print(torch.eq(pred, gt_label))
        
        self.correct += torch.eq(pred_label, gt_label).sum().item()

        self.total += gt_label.numel()
                
        # from mld.data.MOST import sty_name2num_dict
        # labels_act = torch.tensor(list(sty_name2num_dict['most_act'].values()))
        # mask = torch.isin(gt_label, labels_act.cuda())
        # correct_masked = torch.eq(pred_label, gt_label) & ~mask
        # self.correct += correct_masked.sum().item()
        # self.total += (~mask).sum().item()


