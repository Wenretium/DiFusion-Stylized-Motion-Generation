import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric


class DIFUSIONLosses(Metric):
    """
    DIFUSION Loss
    """

    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        self.vae_type = cfg.TRAIN.ABLATION.VAE_TYPE
        self.mode = mode
        self.cfg = cfg
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.stage = cfg.TRAIN.STAGE

        losses = []

        # diffusion loss
        if self.stage in ['diffusion', 'vae_diffusion']:
            # instance noise loss
            losses.append("instt2m_loss")
            losses.append("xt2m_loss")
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # prior noise loss
                losses.append("prior_loss")
            # style
            losses.append("textalign_loss")
            losses.append("classify_loss")
            losses.append("insts2m_loss")
            losses.append("xs2m_loss")

        if self.stage in ['vae', 'vae_diffusion']:
            # reconstruction loss
            losses.append("recons_feature")
            losses.append("recons_verts")
            losses.append("recons_joints")
            losses.append("recons_limb")

            losses.append("gen_feature")
            losses.append("gen_joints")

            # KL loss
            losses.append("kl_motion")
            
        if self.stage in ['style_classifier', 'style_classifier_sra']:
            losses.append("classify_loss")

        if self.stage not in ['vae', 'diffusion', 'vae_diffusion', 'style_classifier', 'style_classifier_sra']:
            raise ValueError(f"Stage {self.stage} not supported")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses
        
        cfg.LOSS.LAMBDA_DENOISE = cfg.LOSS.LAMBDA_DENOISE if "LAMBDA_DENOISE" in cfg.LOSS else 1.0

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'instt2m':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_DENOISE
            elif loss.split('_')[0] == 'xt2m':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_DENOISE
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            if loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            elif loss.split('_')[0] == 'textalign':
                # self._losses_func[loss] = torch.nn.MSELoss(reduction='mean')
                if self.cfg.LOSS.TM_ALIGNMENT_TYPE == 'T2M':
                    self._losses_func[loss] = T2M_euclidean_distance_matrix()
                self._params[loss] = cfg.LOSS.LAMBDA_TEXTALIGN
            elif loss.split('_')[0] == 'insts2m':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'xs2m':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'classify':
                self._losses_func[loss] = nn.CrossEntropyLoss()
                self._params[loss] = cfg.LOSS.LAMBDA_CLASSIFY
            else:
                ValueError("This loss is not recognized.")
            if loss.split('_')[-1] == 'joints':
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT

    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("recons_joints", rs_set['joints_rst'],
                                       rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])
        
        if self.stage in ["diffusion", "vae_diffusion"]:

            if self.cfg.condition in ['text_stylelabel']:
                if self.cfg.LOSS.LAMBDA_DENOISE: 
                    # predict noise
                    if self.predict_epsilon:
                        total += self._update_loss("instt2m_loss", rs_set['t2m']['noise_pred'],
                                                rs_set['t2m']['noise'])
                        total += self._update_loss("insts2m_loss", rs_set['s2m']['noise_pred'],
                                                rs_set['s2m']['noise'])
                    # predict x
                    else:
                        total += self._update_loss("xt2m_loss", rs_set['t2m']['pred'],
                                                rs_set['t2m']['latent'])
                        total += self._update_loss("xs2m_loss", rs_set['s2m']['pred'],
                                                rs_set['s2m']['latent'])
                    
                if self.cfg.TRAIN.DO_FORWARD: 
                    # forward
                    if self.cfg.LOSS.TM_ALIGNMENT_TYPE in ['T2M']:
                        total += self._update_loss_textalign("textalign_loss", rs_set['forward']['forward_motion_emb'],
                            rs_set['forward']['forward_text_emb'])
                    if self.cfg.LOSS.LAMBDA_CLASSIFY:
                        total += self._update_loss("classify_loss", rs_set['forward']['pred_label'], rs_set['forward']['gt_label'])


        if self.stage in ["vae_diffusion"]:
            # loss
            # noise+text_emb => diff_reverse => latent => decode => motion
            total += self._update_loss("gen_feature", rs_set['gen_m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("gen_joints", rs_set['gen_joints_rst'],
                                       rs_set['joints_ref'])
            
        if self.stage in ["style_classifier", "style_classifier_sra"]:
            # pred_label = rs_set['pred_label'].argmax(1)
            # accuracy = torch.sum(pred_label == rs_set['gt_label'])/pred_label.shape[0]
            # loss_debug = self._update_loss("classify_loss", rs_set['pred_label'], rs_set['gt_label']).cpu().detach().tolist()
            # print('debug', accuracy, loss_debug)
            total += self._update_loss("classify_loss", rs_set['pred_label'], rs_set['gt_label'])
            

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss
    
    def _update_loss_textalign(self, loss: str, motion_emb, text_emb):
        val = self._losses_func[loss](motion_emb, text_emb)
        getattr(self, loss).__iadd__(val.detach())
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"

class T2M_euclidean_distance_matrix():

    def __init__(self):
        pass

    def __call__(self, matrix1, matrix2):
        """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
        """
        assert matrix1.shape[1] == matrix2.shape[1]
        d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
        d2 = torch.sum(torch.square(matrix1), axis=1,
                    keepdims=True)  # shape (num_test, 1)
        d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
        dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
        
        dists = dists.nan_to_num()
        matching_scores = dists.trace() # 迹: 对角线元素的和
        bs = matrix1.shape[0]
        matching_scores /= bs
        return matching_scores # 越低越好

