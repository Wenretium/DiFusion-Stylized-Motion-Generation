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


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    tasks:
         1 text+style 2 motion
         2 random sampling
         3 reconstruction
    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME

    task = cfg.DEMO.TASK
    style_name = cfg.DEMO.STYLE if cfg.DEMO.STYLE else "Neutral"
    if task == "interpolate":
        alpha = cfg.DEMO.ALPHA if cfg.DEMO.ALPHA!=None else 0.5
        cfg.model.denoiser.params.interpolate = True
        cfg.model.denoiser.params.interpolate_alpha = alpha
        assert cfg.DEMO.STYLE1 and cfg.DEMO.STYLE2, 'you must specify both style1 and style2 in interpolate mode.'
        style_name1 = cfg.DEMO.STYLE1
        style_name2 = cfg.DEMO.STYLE2

    if task == 'textstyle2motion':
        if cfg.DEMO.EXAMPLE:
            # Check txt file input
            # load txt
            from mld.utils.demo_utils import load_example_input
            text, length = load_example_input(cfg.DEMO.EXAMPLE)
        else:
            # keyborad input
            text = input("Please enter texts, none for random latent sampling:")
            length = input(
                "Please enter length, range 16~196, e.g. 50, none for random latent sampling:"
            )
            # default lengths
            length = 200 if not length else length
            length = [int(length)]
            text = [text]

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
        # batch

        if task == 'textstyle2motion':
            # with style
            condition = cfg.condition
            assert condition in ['text_stylelabel'], 'only support text+stylelabel condition in our released project for simplicity.'
            # use motion or label as style reference
            use_style_motion = False
            if use_style_motion:
                # get style reference motion 
                name = f"{style_name}_FR_1000_1140"
                motion_sty = np.load(style_motion_dir + name + ".npy")
                # get style name from this motion via a pretrained style classifier
                mean = np.load('./deps/t2m/hm_style100/Mean.npy')
                std = np.load('./deps/t2m/hm_style100/Std.npy')
                motion_sty = (motion_sty - mean) / std
                motion_sty = motion_sty.astype(np.float32)
                motion_sty = torch.tensor(motion_sty).unsqueeze(0).to(device)
                style_label = model.get_style_label_from_motion(motion_sty).cpu().numpy().tolist()[0]
                style_name = sty_num2name[style_label]
                print('predict style as', style_name)
            else: # use the input style name
                assert style_name in sty_name2num, 'style name is invalid.'
                style_label = sty_name2num[style_name]
            # preprocess for batch
            bs = len(text)
            style = torch.full((bs, 1), style_label).to(device) 
            # try w/o style or text, default both exist        
            w_text = 1
            w_style = 1
            if not w_style: # w/o style
                style = torch.full((bs, 1), 51).to(device)
                style_name = ''
            if not w_text: # w/o text
                text = [f"{i}" for i in range(bs)]
            batch = {"length": length, "text": text, "style": style}

            if w_style and w_text:
                new_folder = "{}_{}+{}".format(epoch_num, style_name, 'text')
            elif w_style:
                new_folder = "{}_{}".format(epoch_num, style_name)
            elif w_text:
                new_folder = "{}_{}".format(epoch_num, 'text')
                
        elif task == 'interpolate':
            # with style
            condition = cfg.condition
            assert condition in ['text_stylelabel'], 'only support text+stylelabel condition in our released project for simplicity.'
            assert style_name1 in sty_name2num, 'style name1 is invalid.'
            assert style_name2 in sty_name2num, 'style name2 is invalid.'
            style_label1, style_label2 = sty_name2num[style_name1], sty_name2num[style_name2]
                
            bs = len(text)
            style1 = torch.full((bs, 1), style_label1).to(device)
            style2 = torch.full((bs, 1), style_label2).to(device)
            batch = {"length": length, "text": text, "style": [style1, style2]}

            new_folder = "{}_{}-{}-{}+{}".format(epoch_num, style_name1, alpha, style_name2, 'text')

        elif task == 'random_sampling':
            length = 196
            nsample, latent_dim = 20, 256
            batch = {
                "latent":
                torch.randn(1, nsample, latent_dim, device=model.device),
                "length": [int(length)] * nsample,
                "text": ["random sampling"] * nsample,
            }
            new_folder = "{}_{}".format(epoch_num, 'random_sampling')

        elif task == "reconstruction":
            joints = []
            length = []
            motions = [text_motion_dir + '/000000.npy',text_motion_dir + '/000200.npy']
            mean = np.load('./deps/t2m/hm_style100/Mean.npy')
            std = np.load('./deps/t2m/hm_style100/Std.npy')
            for _, motion in enumerate(motions):
                motion = np.load(motion)
                leng = len(motion)
                motion = (motion - mean) / std
                motion = torch.from_numpy(motion).to(torch.float32).to(device)
                motion = motion.unsqueeze(0)
                batch = {"motion": motion, "length": [leng]}
                jots, _ = model.recon_from_motion(batch)
                joints.extend(jots)
                length.append(leng)

            batch["text"] = ["reconstruction"] * len(joints)
            new_folder = "{}_{}".format(epoch_num, 'recon')

        else:
            raise ValueError(
                f"Not support task {task}, only support random_sampling, reconstruction, text+style2motion"
            )

        output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), new_folder))
        output_dir.mkdir(parents=True, exist_ok=True)

        for rep in range(cfg.DEMO.REPLICATION):
            if task == 'random_sampling':
                joints = model.gen_from_latent(batch)
            elif task == 'reconstruction':
                pass
            else:
                # conditioned motion synthesis
                joints = model(batch)

            # cal inference time
            infer_time = time.time() - mld_time
            num_batch = 1
            num_all_frame = sum(batch["length"])
            num_ave_frame = sum(batch["length"]) / len(batch["length"])

            nsample = len(joints)
            for i in range(nsample):
                if task == "random_sampling":
                    motion_name = f"random_sampling_{length}_{i}"
                elif task == "reconstruction":
                    motion_name = f"recon_{length[i]}_{i}"
                elif task == "interpolate":
                    motion_name = f"{style_name1}-{alpha}-{style_name2}+{text[i].replace(' ', '_').replace('.', '')}"
                else:
                    motion_name = f"{style_name}+{text[i].replace(' ', '_').replace('.', '')}"
                npypath = str(output_dir / f"{motion_name}_rep{rep}.npy")
                with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                    text_file.write(batch["text"][i])
                np.save(npypath, joints[i].detach().cpu().numpy())
                print(f"Motions are generated here:\n{npypath}")
            
            rep_lst.append(joints)
            texts_lst.append(batch["text"])
                    

        total_time = time.time() - total_time
        print(f'MLD Infer time - This/Ave batch: {infer_time/num_batch:.3f}')
        print(f'MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(
            f'MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')
        print(
            f'MLD Infer FPS - {num_all_frame/infer_time:.2f}s')
        print(
            f'MLD Infer FPS - time for 100 Poses: {infer_time/(num_batch*num_ave_frame)*100:.2f}'
        )
        print(
            f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
        )

    if cfg.DEMO.RENDER:
        from mld.data.humanml.utils.plot_script import plot_3d_motion

        if task == "random_sampling":
            for i in range(nsample):
                npypath = output_dir / \
                        f"random_sampling_{length}_{i}.npy"
                fig_path = Path(str(npypath).replace(".npy",".mp4"))
                plot_3d_motion(fig_path, paramUtil.t2m_kinematic_chain, joints[i].detach().cpu().numpy(), title="random_sampling", fps=cfg.DEMO.FRAME_RATE)
        elif task == "reconstruction":
            for i in range(nsample):
                npypath = output_dir / \
                        f"recon_{length[i]}_{i}.npy"
                fig_path = Path(str(npypath).replace(".npy",".mp4"))
                plot_3d_motion(fig_path, paramUtil.t2m_kinematic_chain, joints[i].detach().cpu().numpy(), title="reconstruction", fps=cfg.DEMO.FRAME_RATE)
        elif task == "interpolate":
            for i in range(nsample):
                for rep in range(cfg.DEMO.REPLICATION):
                    motion_name = f"{style_name1}-{alpha}-{style_name2}+{text[i].replace(' ', '_').replace('.', '')}"
                    npypath = str(output_dir / f"{motion_name}_rep{rep}.npy")
                    fig_path = Path(str(npypath).replace(".npy",".mp4"))
                    new_title = f"{style_name1}-{alpha}-{style_name2} + {text[i]}"
                    plot_3d_motion(fig_path, paramUtil.t2m_kinematic_chain, rep_lst[rep][i].detach().cpu().numpy(), title=new_title, fps=cfg.DEMO.FRAME_RATE)
        elif text:
            for i in range(nsample):
                for rep in range(cfg.DEMO.REPLICATION):
                    motion_name = f"{style_name}+{text[i].replace(' ', '_').replace('.', '')}"
                    npypath = str(output_dir / f"{motion_name}_rep{rep}.npy")
                    fig_path = Path(str(npypath).replace(".npy",".mp4"))
                    new_title = "{} + {}".format(style_name, text[i])
                    plot_3d_motion(fig_path, paramUtil.t2m_kinematic_chain, rep_lst[rep][i].detach().cpu().numpy(), title=new_title, fps=cfg.DEMO.FRAME_RATE)
            

if __name__ == "__main__":
    main()
