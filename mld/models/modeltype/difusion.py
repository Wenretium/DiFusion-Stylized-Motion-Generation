import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import random
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import torch.nn.functional as F
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    t2m_motionenc,
    t2m_textenc,
)
from mld.models.losses.difusion import DIFUSIONLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding
from mld.data.DiFusion import sty_name2num_dict

from .base import BaseModel


class DIFUSION(BaseModel):
    """
    Stage 1 vae, diffusion
    Stage 2 stage2
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        self.vae_type = cfg.model.vae_type
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        
        if "textencoder_type" not in cfg.model:
            cfg.model.textencoder_type = "CLIP"
        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        if self.stage == "diffusion" and self.cfg.TRAIN.DO_FORWARD:
            self.stage = "stage2"
        else:
            self.TM_alignment_type = "" # 不是stage2则此设置无效

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)
        self.inversion_scheduler = instantiate_from_config(cfg.model.inversion_scheduler)
        
        try:
            self.TM_alignment_type = self.cfg.LOSS.TM_ALIGNMENT_TYPE
        except:
            self.TM_alignment_type = "T2M"
                    
        self.sty_name2num = sty_name2num_dict[cfg.TRAIN.DATASETS[0].lower()]
        self.sty_num2name = dict(zip(self.sty_name2num.values(), self.sty_name2num.keys()))
        self.Neutral_label = self.sty_name2num["Neutral"]
        self.num_styles = len(self.sty_name2num.keys())
        # print('self.Neutral_label', self.Neutral_label)

        from mld.models.architectures.mld_style_encoder import StyleClassification
        if self.stage == 'style_classifier':
            self.style_classifier = StyleClassification(cfg.NUM_STYLES)
        elif 'FLAG' in cfg.METRIC and cfg.METRIC.FLAG == 'TESTSET1': # eval
            if cfg.TESTSET1.NUM_STYLES == 100:
                self.style_classifier = StyleClassification(nclasses=47)
            elif cfg.TESTSET1.NUM_STYLES == 16:
                self.style_classifier = StyleClassification(nclasses=16)
        elif "Name" not in cfg or cfg.Name[:4] == 'demo': # demo
            self.style_classifier = StyleClassification(cfg.TESTSET1.NUM_STYLES)
        
        if self.stage in ['style_classifier']:
            # freeze vae and denoiser
            self.vae.eval()
            self.denoiser.eval()
            for p in self.vae.parameters():
                p.requires_grad = False
            for p in self.denoiser.parameters():
                p.requires_grad = False
        else:
            # freeze style_classifier
            self.style_classifier.eval()
            for p in self.style_classifier.parameters():
                p.requires_grad = False
            if self.stage == "diffusion":
                # freeze vae
                if self.vae_type in ["mld", "vposert","actor"]:
                    self.vae.training = False
                    for p in self.vae.parameters():
                        p.requires_grad = False
                        

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters(), weight_decay=0.01)
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "difusion":
            self._losses = MetricCollection({
                split: DIFUSIONLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "model only supports difusion losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        if self.stage == "stage2": # stage2 为了节省显存，不进行验证
            self.metrics_dict = []
        # 量化对比实验
        if "FLAG" not in cfg.METRIC:
            self.cfg.METRIC.FLAG = ""
        elif self.cfg.METRIC.FLAG == "TESTSET1":
            self.metrics_dict = ['StyMoContMetrics', 'ClassifyMetrics']
        self.configure_metrics()
            
        # Load Evaluation Models
        if "StyMoContMetrics" in self.metrics_dict or self.TM_alignment_type == 'T2M':
            self._get_t2m_evaluator(cfg)

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond', "text_stylecode", "text_stylelabel", "text_stylename", "stylelabel"]:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrained
        dataname = "t2m"
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

   
    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch, return_latent_code=False, task=None, doubletake=False):
        texts = batch["text"]
        lengths = batch["length"]
        styles = batch["style"]
    
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion', 'stage2']:
            # compute cond_emb
            if self.condition in ['text', 'text_uncond']:
                if self.do_classifier_free_guidance:
                    uncond_tokens = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                text_emb = self.text_encoder(texts)
                cond_emb = {"text_emb": text_emb}
            elif 'style' in self.condition:
                # text embs
                if self.do_classifier_free_guidance:
                    uncond_tokens = [""] * len(texts)
                    uncond_tokens.extend(texts)
                    texts = uncond_tokens
                text_emb = self.text_encoder(texts)
                if self.cfg.model.textencoder_type == "TMA":
                    text_emb = text_emb.loc.unsqueeze(1)
                # style embs
                elif self.condition in ['text_stylelabel']:
                    style_emb = styles
                if self.do_classifier_free_guidance:
                    if type(style_emb) == list: # interpolate
                        style_emb = [torch.cat((style_emb[0], style_emb[0]), dim=0),
                                    torch.cat((style_emb[1], style_emb[1]), dim=0)]
                    else:
                        style_emb = torch.cat((style_emb, style_emb), dim=0)
                        
                cond_emb = {"text_emb": text_emb, "style_emb": style_emb}
            
            if task == 'style_transfer':
                motions = batch["motion"]
                
                latents, _ = self.vae.encode(motions, lengths)
                # add noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each motion
                timesteps = torch.full((bsz, ), self.noise_scheduler.config.num_train_timesteps)
                # timesteps = torch.full((bsz, ), 0)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                latents = self.noise_scheduler.add_noise(latents.clone(), noise, timesteps)

                z = self._diffusion_reverse(cond_emb, lengths, latents)
            else:
                z = self._diffusion_reverse(cond_emb, lengths)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')

        if return_latent_code:
            return feats_rst.detach().cpu()
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, latents=None):
        # init latents
        bsz = encoder_hidden_states["text_emb"].shape[0] # encoder_hidden_states is a dict
        device = encoder_hidden_states["text_emb"].device
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if latents == None: # random sample
            if self.vae_type == "no":
                assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
                latents = torch.randn(
                    (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                    device=device,
                    dtype=torch.float,
                )
            else:
                latents = torch.randn(
                    (bsz, self.latent_dim[0], self.latent_dim[-1]),
                    device=device,
                    dtype=torch.float,
                )
        # print('latents', self.do_classifier_free_guidance, latents.shape)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents
    

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,                                                  # low
            self.noise_scheduler.config.num_train_timesteps,    # high
            (bsz, ),                                            # size
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if "text" in self.condition:
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def train_diffusion_forward(self, batch):
        t2m_motion = batch["t2m_motion"]
        t2m_style = batch["t2m_style"]
        t2m_m_length = batch["t2m_m_length"]
        t2m_style_label = batch["t2m_style_label"]
        s2m_motion = batch["s2m_motion"]
        s2m_style = batch["s2m_style"]
        s2m_style_label = batch["s2m_style_label"]
        s2m_style_label_name = batch["s2m_style_label_name"]
        s2m_m_length = batch["s2m_m_length"]

        # motion encode
        with torch.no_grad():
            t2m_z, _ = self.vae.encode(t2m_motion)
            s2m_z, _ = self.vae.encode(s2m_motion)
            
        if self.condition in ["text_stylelabel", "text_stylename", "text_stylecode"]:
            # t2m
            t2m_text = batch["t2m_text"]
            t2m_text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in t2m_text
            ]
            t2m_text_emb = self.text_encoder(t2m_text)
            # s2m
            s2m_text = batch["s2m_text"]
            s2m_text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in s2m_text
            ]
            s2m_text_emb = self.text_encoder(s2m_text)
            
            if self.cfg.model.textencoder_type == "TMA":
                t2m_text_emb = t2m_text_emb.loc.unsqueeze(1)
                s2m_text_emb = s2m_text_emb.loc.unsqueeze(1)
                
            if self.condition == "text_stylelabel":
                t2m_style_emb = t2m_style_label
                s2m_style_emb = s2m_style_label
                # print('style_label', t2m_style_label, s2m_style_label)
            elif self.condition == "text_stylename": # use CLIP to encode style name instead of using labels
                t2m_style_emb = self.text_encoder(["Neutral"] * len(t2m_text))
                s2m_style_emb = self.text_encoder(s2m_style_label_name)
                # print(t2m_style_emb.shape, s2m_style_emb.shape)
            elif self.condition == "text_stylecode": # use style encoder for style embs
                t2m_style_emb = self.style_encoder(t2m_style, stage="Encode")
                s2m_style_emb = self.style_encoder(s2m_style, stage="Encode")
                # print(s2m_style_label.shape, s2m_style_emb.shape)
                # for i, label in enumerate(t2m_style_label[:,0].detach().tolist()):
                #     # if label == 51:
                #     print(t2m_style.shape, label, t2m_style_emb[i, :2], t2m_style[i, :5, 100])
            
            t2m_cond_emb = {"text_emb": t2m_text_emb, "style_emb": t2m_style_emb}
            s2m_cond_emb = {"text_emb": s2m_text_emb, "style_emb": s2m_style_emb}

            if self.stage == "stage2":
                # forward: compute text-motion alignment loss
                forward_text = batch["t2m_text"]
                if self.do_classifier_free_guidance: # 1/2概率无法同时具有text+style
                    uncond_tokens = [""] * len(forward_text)
                    # uncond_tokens = []
                    # uncond_tokens.extend(forward_text)
                    uncond_tokens.extend(forward_text)
                    forward_text = uncond_tokens
                    # random.shuffle(forward_text)
                    forward_text_emb = self.text_encoder(forward_text)
                    # forward_style_emb = torch.cat((s2m_style_emb, torch.full(s2m_style_label.shape, 51).to(s2m_style_label.device)), dim=0)
                    forward_style_emb = torch.cat((s2m_style_emb, s2m_style_emb), dim=0)
                else:
                    forward_text_emb = self.text_encoder(forward_text)
                    forward_style_emb = s2m_style_emb
                # forward_style_emb = torch.full(forward_style_emb.shape, 51).to(forward_style_emb.device) # 51
                if self.cfg.model.textencoder_type == "TMA":
                    forward_text_emb = forward_text_emb.loc.unsqueeze(1)
                forward_cond = {"text_emb": forward_text_emb, "style_emb": forward_style_emb}
                z = self._diffusion_reverse(forward_cond, t2m_m_length) # 随机设置长度为t2m_m_length
                forward_motion = self.vae.decode(z, t2m_m_length)
                forward_z = z

                # text-motion alignment
                if self.TM_alignment_type == 'T2M':
                    m_lens = t2m_m_length.copy()
                    m_lens = torch.tensor(m_lens, device=forward_motion.device)
                    align_idx = np.argsort(m_lens.data.tolist())[::-1].copy() # 长度从大到小排列的索引
                    forward_motion_ = forward_motion[align_idx] # 长度从大到小排列
                    m_lens = m_lens[align_idx] # 长度从大到小排列
                    m_lens = torch.div(m_lens, # 这个操作不确定要不要保留？？注释掉会报cuda错误
                                    self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                                    rounding_mode="floor")
                    forward_motion_mov = self.t2m_moveencoder(forward_motion_[..., :-4])
                    forward_motion_emb = self.t2m_motionencoder(forward_motion_mov, m_lens)
                    word_embs = batch["t2m_word_embs"]
                    pos_ohot = batch["t2m_pos_ohot"]
                    text_lengths = batch["t2m_text_len"]
                    forward_text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]
                    
                else:
                    raise TypeError(f"text-motion alignment type {self.TM_alignment_type} not supported. The ablation studies were removed from our final released code for simplicity.")
                
                # style classification alignment
                if self.cfg.LOSS.LAMBDA_CLASSIFY:
                    forward_pred_label = self.style_classifier(forward_motion)
                
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        
        if self.condition in ["text_stylelabel"]:
            n_set = {}
            # Noise prediction loss
            # t2m
            n_set["t2m"] = self._diffusion_process(t2m_z, t2m_cond_emb, t2m_m_length)
            
            # s2m
            n_set["s2m"] = self._diffusion_process(s2m_z, s2m_cond_emb, s2m_m_length)

            if self.stage == "stage2":
                # forward
                n_set["forward"] = {
                "forward_motion_emb": forward_motion_emb,
                "forward_text_emb": forward_text_emb
                }
                if self.cfg.LOSS.LAMBDA_CLASSIFY:
                    n_set["forward"].update({
                    "pred_label": forward_pred_label,
                    "gt_label": s2m_style_label.squeeze(dim=1),
                    })
        
        return {**n_set}


    def train_style_classifier(self, batch):
        motion, length, label_sty = batch['motion'], batch['length'], batch['label_sty']
        
        # (bs, T, dims)
        pred_label = self.style_classifier(motion)
        
        gt_label = torch.Tensor(label_sty).squeeze()
        
        classifier_set = {"pred_label": pred_label, "gt_label": gt_label}
        
        return classifier_set
    
    
    def test_diffusion_forward(self, batch, finetune_decoder=False): # 这个函数已经不调用了
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
                
        elif self.condition in ["text_stylelabel"]:
            cond_emb = batch['style_label']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['style_label'],
                                     dtype=batch['style_label'].dtype))
            
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set


    def difusion_eval_stage1(self, batch):
        if self.stage == "vae":
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]
        elif self.stage == "diffusion":
            if "t2m_text" not in batch.keys(): # phase == "testset1": 
                # defined in TextStyle2Motion_Testset1
                texts = batch["text"]
                motions = batch["motion"].detach().clone()
                lengths = batch["length"]
                word_embs = batch["word_embs"].detach().clone()
                pos_ohot = batch["pos_ohot"].detach().clone()
                text_lengths = batch["text_len"].detach().clone()
                style_labels = batch["style"].detach().clone()
            else: 
                texts = batch["t2m_text"]
                motions = batch["t2m_motion"].detach().clone()
                lengths = batch["t2m_m_length"]
                word_embs = batch["t2m_word_embs"].detach().clone()
                pos_ohot = batch["t2m_pos_ohot"].detach().clone()
                text_lengths = batch["t2m_text_len"].detach().clone()
                style_labels = batch["s2m_style_label"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            style_labels = style_labels.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion', 'stage2']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition in ['text', 'text_stylelabel', 'text_stylename', 'text_stylecode']:
                    uncond_tokens.extend(texts)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            
            if self.condition == 'text_stylelabel':
                style_emb = style_labels
            if self.do_classifier_free_guidance:
                style_emb = torch.cat((style_emb, style_emb), dim=0)
            
            if self.cfg.model.textencoder_type == "TMA":
                text_emb = text_emb.loc.unsqueeze(1)
            cond_emb = {"text_emb": text_emb, "style_emb": style_emb}
            z = self._diffusion_reverse(cond_emb, lengths)
            
        elif self.stage in ['vae']:
            z, dist_m = self.vae.encode(motions, lengths)
            
        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # motion_neutral = self.vae.decode(z_neutral, lengths)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        t2m_feats_rst = self.datamodule.renorm4t2m(feats_rst)
        t2m_motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=t2m_motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        t2m_motions = t2m_motions[align_idx]
        t2m_feats_rst = t2m_feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(t2m_feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(t2m_motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        if self.stage != "vae":
            # t2m text encoder
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                            text_lengths)[align_idx]
        
        if self.stage == "vae":
            rs_set = {
                "m_ref": t2m_motions,
                "m_rst": t2m_feats_rst,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst
            }
            return rs_set
        
        # s2m
        # feats_rst -= motion_neutral # 会提升分类准确率
        s2m_pred_label = self.style_classifier(feats_rst)
        s2m_gt_label = style_labels
        

        rs_set = {
            "gen": z.squeeze(0), # (1, bs, 512) --> (bs, 512)
            "t2m_text_emb": text_emb,
            "t2m_gen_emb": recons_emb,
            "t2m_gt_emb": motion_emb,
            "pred_label": s2m_pred_label,
            "gt_label": s2m_gt_label,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set
    

    def difusion_eval_all(self, batch):
        if "t2m_text" not in batch.keys(): # phase == "testset1":
            texts = batch["text"]
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]
            word_embs = batch["word_embs"].detach().clone()
            pos_ohot = batch["pos_ohot"].detach().clone()
            text_lengths = batch["text_len"].detach().clone()
            style_labels = batch["style"].detach().clone()
            # w/o style
            # style_labels = torch.full(style_labels.shape, self.Neutral_label).to(style_labels.device)
            style_names = batch["style_label_name"]
            style_motions = batch["style_motion"].detach().clone()
        elif self.stage in ["diffusion", "stage2"]:
            texts = batch["t2m_text"]
            motions = batch["t2m_motion"].detach().clone()
            lengths = batch["t2m_m_length"]
            word_embs = batch["t2m_word_embs"].detach().clone()
            pos_ohot = batch["t2m_pos_ohot"].detach().clone()
            text_lengths = batch["t2m_text_len"].detach().clone()
            style_labels = batch["s2m_style_label"].detach().clone()
            style_names = batch["s2m_style_label_name"]
            style_motions = batch["s2m_motion"].detach().clone()
        elif self.stage == "vae":
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            style_labels = style_labels.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            style_names = style_names * self.cfg.TEST.MM_NUM_REPEATS
            style_motions = style_motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion', 'stage2']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition in ['text', 'text_style', 'text_stylelabel', 'text_stylename', 'text_stylecode']:
                    uncond_tokens.extend(texts)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            
            if self.condition == 'text_stylelabel':
                style_emb = style_labels
            if self.do_classifier_free_guidance:
                style_emb = torch.cat((style_emb, style_emb), dim=0)
            
            cond_emb = {"text_emb": text_emb, "style_emb": style_emb}
            z = self._diffusion_reverse(cond_emb, lengths)
            
            # Neutral
            if self.condition == 'text_stylelabel':
                style_emb_neutral = torch.full(style_labels.shape, self.Neutral_label).to(style_labels.device)
            if self.do_classifier_free_guidance:
                style_emb_neutral = torch.cat((style_emb_neutral, style_emb_neutral), dim=0)
            
            cond_emb = {"text_emb": text_emb, "style_emb": style_emb_neutral}
            z_neutral = self._diffusion_reverse(cond_emb, lengths)
            
        elif self.stage in ['vae']:
            z, dist_m = self.vae.encode(motions, lengths)
            
        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            motion_neutral = self.vae.decode(z_neutral, lengths)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        t2m_feats_rst = self.datamodule.renorm4t2m(feats_rst)
        t2m_motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=t2m_motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        t2m_motions = t2m_motions[align_idx]
        t2m_feats_rst = t2m_feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(t2m_feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(t2m_motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)
        
        if self.stage != "vae":
            # t2m text encoder
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                            text_lengths)[align_idx]
        
        if self.stage == "vae":
            rs_set = {
                "m_ref": t2m_motions,
                "m_rst": t2m_feats_rst,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst
            }
            return rs_set
        
        # s2m
        feats_rst -= motion_neutral # 对于我们的模型而言，会提升分类准确率
        s2m_pred_label = self.style_classifier(feats_rst)
        s2m_gt_label = style_labels
        if self.cfg.TEST.DATASETS == ['difusion']:
            s2m_pred_label_filter = []
            s2m_gt_label_filter = []
            for i, label_name in enumerate(batch['style_label_name']):
                try: # 舍弃其他的风格，不参与计算
                    gt_label = sty_name2num_dict['difusion_47'][label_name]
                    s2m_gt_label_filter.append(gt_label)
                    s2m_pred_label_filter.append(s2m_pred_label[i].detach().cpu().numpy().tolist())
                except:
                    pass
            s2m_gt_label = torch.as_tensor(s2m_gt_label_filter).unsqueeze(1)
            s2m_pred_label = torch.as_tensor(s2m_pred_label_filter)
        # _, pred_label_print = s2m_pred_label.topk(1, 1, True, True)
        # print('sra', s2m_gt_label[:5], pred_label_print[:5])
        # print('sra', s2m_gt_label.shape, s2m_pred_label.shape)
                  
        rs_set = {
            "gen": z.squeeze(0), # (1, bs, 512) --> (bs, 512)
            "t2m_text_emb": text_emb,
            "t2m_gen_emb": recons_emb,
            "t2m_gt_emb": motion_emb,
            "pred_label": s2m_pred_label,
            "gt_label": s2m_gt_label,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set
    

    def style_classifier_eval(self, batch):
        motion, length, label_sty = batch['motion'], batch['length'], batch['label_sty']
        
        # (bs, T, dims)
        with torch.no_grad():
            pred_label = self.style_classifier(motion)
            gt_label = torch.Tensor(label_sty).squeeze()

        rs_set = {"pred_label": pred_label, "gt_label": gt_label}
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["t2m_motion"].detach().clone()
        lengths = batch["t2m_m_length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["t2m_word_embs"].detach()
        pos_ohot = batch["t2m_pos_ohot"].detach()
        text_lengths = batch["t2m_text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage in ["diffusion", "stage2"]:
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            elif self.stage == "style_classifier":
                rs_set = self.train_style_classifier(batch)
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            # 有些变量名称不一样
            if "t2m_m_length" not in batch.keys():
                batch["t2m_m_length"] = batch["length"]

            # TEST STAGE
            if self.cfg.METRIC.FLAG == "TESTSET1":
                rs_set = self.difusion_eval_all(batch)
            # TRAIN STAGE
            elif self.stage in ['vae', 'diffusion']: # stage1
                rs_set = self.difusion_eval_stage1(batch)
            elif self.stage == "stage2": # stage2 为了节省显存，不进行验证
                re_set = {}
            elif self.stage == "style_classifier":
                rs_set = self.style_classifier_eval(batch)

            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "StyMoContMetrics":
                    getattr(self, metric).update(
                        rs_set["t2m_text_emb"],
                        rs_set["t2m_gen_emb"],
                        rs_set["t2m_gt_emb"],
                        batch["t2m_m_length"],
                    )
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["t2m_text_emb"],
                        rs_set["t2m_gen_emb"],
                        rs_set["t2m_gt_emb"],
                        batch["t2m_m_length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "ReconMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["t2m_m_length"],
                    )
                elif metric == "ClassifyMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["pred_label"], rs_set["gt_label"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            if self.stage in ["style_classifier"]:
                return rs_set
            else:
                return rs_set["joints_rst"], batch["t2m_m_length"]
                # return None # 这里返回啥都行
        return loss

    def get_latent_codes(self, batch):
        # print(batch.keys())
        codes = {}

        if self.condition in ['text_stylelabel']:

            codes["style_code"] = batch["motion"] # 对原风格动作数据降维，发现不具有聚类分布规律
            codes["style_label"] = []
            for label in batch["label_sty"]:
                codes["style_label"].append(self.sty_num2name[label.detach().cpu().tolist()[0]])
            
        return codes
    
    def get_style_label_from_motion(self, motion):
        label_sty = self.style_classifier(motion)
        label_sty = label_sty.argmax(1)
        
        return label_sty

    @torch.no_grad()
    def invert(self, batch):
        texts = batch["text"]
        styles = batch["style"]
        lengths = batch["length"]
        motion = batch["motion"].cuda().float()        

        latents, dist_m = self.vae.encode(motion, lengths)
        # return latents

        # 5. Encode input prompt
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * len(texts)
            uncond_tokens.extend(texts)
            texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            text_emb = text_emb.cuda()

        style_emb = styles
        if self.do_classifier_free_guidance:
            style_emb = torch.cat((style_emb, style_emb), dim=0)
        
        cond_emb = {"text_emb": text_emb, "style_emb": style_emb}

        # 4. Prepare timesteps
        num_inference_steps = self.cfg.model.inversion_scheduler.num_inference_timesteps
        self.inversion_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inversion_scheduler.timesteps

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inversion_scheduler.order
        intermediate_latents = []

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.inversion_scheduler.scale_model_input(latent_model_input, t)

            self.denoiser = self.denoiser.cuda()

            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t.cuda(),
                encoder_hidden_states=cond_emb,
                lengths=lengths,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inversion_scheduler.step(noise_pred, t, latents).prev_sample
            intermediate_latents.append(latents)

        start_step=38
        inverted_latents = intermediate_latents[-(start_step+1)] #latents.detach().clone()

        return inverted_latents


    @torch.no_grad()
    def mst_forward(self, batch, latents):
        texts = batch["text"]
        lengths = batch["length"]     
        styles = batch["style"]
        batch = len(texts)

        start_step=38 # timesteps是倒序的,这里决定从中间哪步开始去噪

        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.cuda()

        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * len(texts)
            uncond_tokens.extend(texts)
            texts = uncond_tokens
            text_emb = self.text_encoder(texts)
        
        style_emb = styles
        if self.do_classifier_free_guidance:
            if type(style_emb) == list: # interpolate
                style_emb = [torch.cat((style_emb[0], style_emb[0]), dim=0),
                            torch.cat((style_emb[1], style_emb[1]), dim=0)]
            else:
                style_emb = torch.cat((style_emb, style_emb), dim=0)
            
        cond_emb = {"text_emb": text_emb, "style_emb": style_emb}
        
        for i in range(start_step, self.cfg.model.scheduler.num_inference_timesteps):
            t = timesteps[i]
                
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=cond_emb,
                lengths=lengths_reverse,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2)

        feats_rst = self.vae.decode(latents, lengths)
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)