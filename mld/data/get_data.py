from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer
from .DiFusion import DIFUSIONDataModule
from .STYLEMO import STYLEMODataModule
from .HumanML3D import HumanML3DDataModule
from .utils import *


def get_mean_std(phase, cfg, dataset_name):

    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit", "difusion", "style100", "hm_bfa", "bfa"]
    # if phase in ["train", "val", "test"]:
    if phase in ["val"]:
        if name in ["t2m", "difusion", "style100", "hm_bfa", "bfa"]:
            data_root = pjoin(cfg.model.t2m_path, "t2m", "Comp_v6_KLD01",
                              "meta")
        elif name == 'kit':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005",
                              "meta")
        else:
            raise ValueError("Only support t2m and kit")
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    elif name in ['difusion', 'style100', 'hm_bfa', 'bfa']:
        data_root = pjoin(cfg.model.t2m_path, "hm_style100")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg, phase, dataset_name):
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit", "difusion", "hm_bfa"]:
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, phase="train", stage="diffusion"):
    if name.lower() in ["humanml3d", "kit"]:
        if phase == "testset1":
            return mld_collate_w_style
        else:
            return mld_collate
    elif name.lower() in ["difusion", "hm_bfa"]:
        if phase == "testset1":
            return mld_collate_w_style
        elif stage == "vae":
            return most_collate_vae
        elif stage == "diffusion":
            return most_collate_diffusion
    elif name.lower() in ["style100", "bfa"]:
        return stylemo_collate
    else:
        raise NotImplementedError

# map config name to module&path
dataset_module_map = {
    "difusion": DIFUSIONDataModule,
    "style100": STYLEMODataModule,
    "hm_bfa": DIFUSIONDataModule,
    "bfa": STYLEMODataModule,
    "humanml3d": HumanML3DDataModule,
}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs", 
                 "difusion": "new_joint_vecs", "style100": "new_joint_vecs", 
                 "hm_bfa": "new_joint_vecs", "bfa": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            text_dir = pjoin(data_root, "texts")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                phase=phase,
                dataset_name=dataset_name,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                visualize=cfg.VISUALIZE,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=text_dir,
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
        elif dataset_name.lower() in ["difusion", "hm_bfa"]:
            # get stage
            stage = cfg.TRAIN.STAGE
            # 
            data_root_t2m = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT_T2M")
            data_root_s2m = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT_S2M")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase, stage)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                stage=stage,
                phase=phase,
                dataset_name=dataset_name,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                visualize=cfg.VISUALIZE,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                motion_dir_t2m=pjoin(data_root_t2m, motion_subdir[dataset_name]),
                motion_dir_s2m=pjoin(data_root_s2m, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
        elif dataset_name.lower() in ["style100", "bfa"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                dataset_name=dataset_name,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                visualize=cfg.VISUALIZE,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
            )
            datasets.append(dataset)
    
    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
