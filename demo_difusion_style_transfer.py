import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
from os.path import join as pjoin

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import mld.data.humanml.utils.paramUtil as paramUtil
from mld.data.DiFusion import sty_name2num_dict
from mld.data.humanml.scripts.motion_process import recover_from_ric

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    current tasks:
        motion style transfer
    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME

    task = "style_transfer"

    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, phase="test")[0]

    # create mld model
    total_time = time.time()
    model = get_model(cfg, dataset)

    # loading checkpoints
    print("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    if '=' in cfg.TEST.CHECKPOINTS:
        epoch_num = cfg.TEST.CHECKPOINTS.split('=')[1].split('.')[0]
    else:
        epoch_num = 'best'
    
    print("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    sty_name2num = sty_name2num_dict[cfg.TEST.DATASETS[0]]
    sty_num2name = dict(zip(sty_name2num.values(), sty_name2num.keys()))

    mld_time = time.time()

    # prepare data dir
    text_motion_dir = cfg.DATASET.DIFUSION.ROOT_T2M + '/new_joint_vecs/'
    style_motion_dir = cfg.DATASET.DIFUSION.ROOT_S2M + '/new_joint_vecs/'

    # sample
    with torch.no_grad():
        rep_lst = []    
        rep_ref_lst = []
        texts_lst = []
        # content reference motion
        content_name = cfg.DEMO.CONTENT
        with open(text_motion_dir.replace("new_joint_vecs", "texts") + content_name + ".txt") as f:
            line = f.readlines()[0]
            line_split = line.strip().split("#")
            text = line_split[0]
        content_motion = np.load(text_motion_dir + content_name + ".npy")
        length = len(content_motion)
        mean = np.load('./deps/t2m/hm_style100/Mean.npy')
        std = np.load('./deps/t2m/hm_style100/Std.npy')
        motion = (content_motion - mean) / std
        motion = motion.astype(np.float32)
        motion = torch.tensor(motion).unsqueeze(0).to(device)

        style_name = cfg.DEMO.STYLE if cfg.DEMO.STYLE else "ArmsFolded"
        assert style_name in sty_name2num, 'style name is invalid.'
        style_label = sty_name2num[style_name]

        nsample = 1
        motion = torch.repeat_interleave(motion, nsample, dim=0)
        length = [length]*nsample
        text = [text]*nsample
        style = torch.full((nsample, 1), style_label).to(device)
        # DDIM inversion
        style_neutral = torch.full((nsample, 1), sty_name2num["Neutral"]).to(device)
        batch = {"length": length, "motion": motion, "text": text, "style": style_neutral}
        inverted_latents = model.invert(batch)

        # forward generation
        batch = {"length": length, "text": text, "style": style}

        new_folder = "ST_{}_{}_{}".format(epoch_num, style_name, content_name)
        output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), new_folder))

        output_dir.mkdir(parents=True, exist_ok=True)

        for rep in range(cfg.DEMO.REPLICATION):
            # inverted_latents = torch.randn_like(inverted_latents)
            joints = model.mst_forward(batch, inverted_latents)
            
            nsample = len(joints)
            for i in range(nsample):
                npypath = output_dir / f"{content_name}-{style_name}_{i}.npy"
                np.save(npypath, joints[i].detach().cpu().numpy())
                print(f"Motions are generated here:\n{npypath}")
            
            rep_lst.append(joints)
            texts_lst.append(batch["text"])
                    

        if task not in ['style_transfer']:
            raise ValueError(
                f"Not support task {task}, this script only support style_transfer"
            )

    if cfg.DEMO.RENDER:
        from mld.data.humanml.utils.plot_script import plot_3d_motion
        if task == "style_transfer":
            for i in range(nsample):
                # transfer result
                fig_path = Path(str(npypath).replace(".npy",".mp4"))
                plot_3d_motion(fig_path, paramUtil.t2m_kinematic_chain, joints[i].detach().cpu().numpy(), title=text[i], fps=cfg.DEMO.FRAME_RATE)
                # content motion
                content_motion_joints = recover_from_ric(torch.from_numpy(content_motion).unsqueeze(0).float(), joints_num=22)
                content_motion_joints = np.array(content_motion_joints[0].detach().cpu())
                fig_path = output_dir /  f"{content_name}.mp4"
                plot_3d_motion(fig_path, paramUtil.t2m_kinematic_chain, content_motion_joints, title=text[i], fps=cfg.DEMO.FRAME_RATE)
                npypath = output_dir /  f"{content_name}.npy"
                np.save(npypath, content_motion_joints)

if __name__ == "__main__":
    main()
