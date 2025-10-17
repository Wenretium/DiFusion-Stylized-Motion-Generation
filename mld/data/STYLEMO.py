import torch
import random
import numpy as np
from .base import BASEDataModule
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
import codecs as cs
from mld.data.DiFusion import sty_name2num_dict
from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

class STYLEMODataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        if cfg.TRAIN.DATASETS == ['style100']:
            self.name = "style100"
        elif cfg.TRAIN.DATASETS ==  ['bfa']:
            self.name = "bfa"
        self.njoints = 22
        self.Dataset = StylizedMotionDataset
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)


class StylizedMotionDataset(data.Dataset):
    
    def __init__(
        self,
        cfg,
        phase,
        dataset_name,
        mean,
        std,
        split_file,
        motion_dir,
        **kwargs,
    ):
        self.cfg = cfg
        self.phase = phase
        self.dataset_name = dataset_name
        '''
        100STYLE: text + motion_cont + motion_sty
        '''
        self.sty_name2num = sty_name2num_dict[self.dataset_name]
        data_dict = {}
        id_list = []
        print('split_file', split_file)
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        enumerator = enumerate(
            track(
                id_list,
                f"Loading 100STYLE training data",
            ))

        name_list = []
        for i, name in enumerator:
            motion_sty = np.load(pjoin(motion_dir, name + ".npy"))
            label_sty, label_cont, _, _ = name.split('_')
            data_dict[name] = {
                    "motion": motion_sty,
                    "length": len(motion_sty),
                    "label_sty": label_sty,
                    "label_cont": label_cont,
                    }
            name_list.append(name)
        
        self.name_list = name_list
        self.data_dict = data_dict
        self.mean = mean
        self.std = std
        self.nfeats = motion_sty.shape[1]
        
        # 计算SRA时剔除一些类别
        if cfg.model.num_styles == 47:
            # 更新剔除most_act后的数据项
            filtered_data_dict = {}
            filtered_name_list = []
            for name in name_list:
                if data_dict[name]['label_sty'] in sty_name2num_dict['difusion_47'].keys():
                    filtered_data_dict[name] = data_dict[name]
                    filtered_name_list.append(name)
            self.data_dict = filtered_data_dict
            self.name_list = filtered_name_list
            self.sty_name2num = sty_name2num_dict['difusion_47']
        
        print('Style num of data' , len(self.name_list))
    
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, length, label_sty_name = data['motion'], data['length'], data['label_sty']
        motion = (motion - self.mean) / self.std
        label_sty = self.sty_name2num[label_sty_name]
        
        return (
            motion,
            length,
            label_sty
        )

            