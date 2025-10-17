from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
import torch.nn.functional as F
import random

from .utils import *

class StyMoContMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "StyMoContMetrics"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []

        #############
        #    t2m    #
        #############

        # Matching scores
        self.add_state("t2m_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("t2m_gt_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.t2m_Matching_metrics = ["t2m_Matching_score", "t2m_gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"t2m_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.t2m_Matching_metrics.append(f"t2m_R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"t2m_gt_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.t2m_Matching_metrics.append(f"t2m_gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.t2m_Matching_metrics)

        # Fid
        self.add_state("t2m_FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("t2m_FID")
        
        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
        self.add_state("t2m_text_emb", default=[], dist_reduce_fx=None)
        self.add_state("t2m_gen_emb", default=[], dist_reduce_fx=None)
        self.add_state("t2m_gt_emb", default=[], dist_reduce_fx=None)

        self.add_state("skate_ratio", default=torch.tensor(0.0), dist_reduce_fx="sum")


    def compute(self, sanity_flag):
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_t2m_text_emb = torch.cat(self.t2m_text_emb,
                              axis=0).cpu()[shuffle_idx, :]
        all_t2m_gen_emb = torch.cat(self.t2m_gen_emb,
                                   axis=0).cpu()[shuffle_idx, :]
        all_t2m_gt_emb = torch.cat(self.t2m_gt_emb,
                                  axis=0).cpu()[shuffle_idx, :]

        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_t2m_text_emb[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_t2m_gen_emb[i * self.R_size:(i + 1) *
                                           self.R_size]
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            # print(dist_mat[:5])
            self.t2m_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        metrics["t2m_Matching_score"] = self.t2m_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"t2m_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_t2m_text_emb[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_t2m_gt_emb[i * self.R_size:(i + 1) *
                                          self.R_size]
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            # match score
            self.t2m_gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        metrics["t2m_gt_Matching_score"] = self.t2m_gt_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"t2m_gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_t2m_gen_emb = all_t2m_gen_emb.numpy()
        all_t2m_gt_emb = all_t2m_gt_emb.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_t2m_gen_emb)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_t2m_gt_emb)
        metrics["t2m_FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        # print("Diversity", all_t2m_gen_emb.shape) # (test_dataset_num, 512)
        metrics["Diversity"] = calculate_diversity_np(all_t2m_gen_emb,
                                                      self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(
            all_t2m_gt_emb, self.diversity_times)
        
        # # Compute Foot Skating Ratio
        # metrics["skate_ratio"] = self.skate_ratio / self.count_seq
        

        return {**metrics}

    def update(
        self,
        t2m_text_emb: Tensor,
        t2m_gen_emb: Tensor,
        t2m_gt_emb: Tensor,
        lengths: List[int],
    ):
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        t2m_text_emb = torch.flatten(t2m_text_emb, start_dim=1).detach()
        t2m_gen_emb = torch.flatten(t2m_gen_emb, start_dim=1).detach()
        t2m_gt_emb = torch.flatten(t2m_gt_emb, start_dim=1).detach()

        # store all texts and motions
        self.t2m_text_emb.append(t2m_text_emb)
        self.t2m_gen_emb.append(t2m_gen_emb)
        self.t2m_gt_emb.append(t2m_gt_emb)

        # skate_ratio, skate_vel = calculate_skating_ratio(motion)
        # self.skate_ratio += skate_ratio


# # from Guided Motion Diffusion for Controllable Human Motion Synthesis
# from scipy.ndimage import uniform_filter1d
# def calculate_skating_ratio(motions):
#     thresh_height = 0.05 # 10
#     fps = 20.0
#     thresh_vel = 0.50 # 20 cm /s 
#     avg_window = 5 # frames

#     batch_size = motions.shape[0]
#     # 10 left, 11 right foot. XZ plane, y up
#     # motions [bs, 22, 3, max_len]
#     verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
#     verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
#     # [bs, 2, max_len-1]
#     vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

#     verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
#     # If feet touch ground in agjecent frames
#     feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
#     # skate velocity
#     skate_vel = feet_contact * vel_avg

#     # it must both skating in the current frame
#     skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
#     # and also skate in the windows of frames
#     skating = np.logical_and(skating, (vel_avg > thresh_vel))

#     # Both feet slide
#     skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
#     skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
#     return skating_ratio, skate_vel