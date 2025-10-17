import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule
import random
import codecs as cs
from torch.utils import data
from rich.progress import track
import os
from os.path import join as pjoin

class DIFUSIONDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 phase,
                 dataset_name,
                 stage,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.njoints = 22
        self.phase = phase
        self.dataset_name = dataset_name
        if phase == "testset1":
            self.Dataset = TextStyle2Motion_Testset1
        elif stage == 'vae':
            self.Dataset = MotionDataset
        elif stage == 'diffusion':
            self.Dataset = TextStyle2MotionDatasetV1
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
            
    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        split_file_t2m = pjoin(
            eval(f"self.cfg.DATASET.{self.dataset_name.upper()}.SPLIT_ROOT_T2M"),
            self.cfg.EVAL.SPLIT + ".txt",
        )
        split_file_s2m = pjoin(
            eval(f"self.cfg.DATASET.{self.dataset_name.upper()}.SPLIT_ROOT_S2M"),
            self.cfg.EVAL.SPLIT + ".txt",
        )
        return self.Dataset(split_file_t2m=split_file_t2m, split_file_s2m=split_file_s2m, **sample_params)
    
    def __getattr__(self, item):
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                subset = subset.upper() if subset != "val" else "EVAL"
                split_file_t2m = pjoin(
                    eval(f"self.cfg.DATASET.{self.dataset_name.upper()}.SPLIT_ROOT_T2M"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt",
                )
                split_file_s2m = pjoin(
                    eval(f"self.cfg.DATASET.{self.dataset_name.upper()}.SPLIT_ROOT_S2M"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt",
                )
                self.__dict__[item_c] = self.Dataset(split_file_t2m=split_file_t2m,
                                                    split_file_s2m=split_file_s2m,
                                                    **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")


class MotionDataset(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file_t2m,
        split_file_s2m,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir_t2m,
        motion_dir_s2m,
        tiny=False,
        debug=False,
        visualize=False,
        progress_bar=True,
        **kwargs,
    ):
        '''
        HumanML3D: text + motion
        '''
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        print('dir_t2m', motion_dir_t2m, split_file_t2m)
        print('dir_s2m', motion_dir_s2m, split_file_s2m)
        text_dir_t2m = pjoin(os.path.dirname(motion_dir_t2m), 'texts')
        text_dir_s2m = pjoin(os.path.dirname(motion_dir_s2m), 'texts')
        
        data_dict = {}
        id_list = []
        text_list = []
        with cs.open(split_file_t2m, "r") as f:
            for line in f.readlines():
                id_list.append(pjoin(motion_dir_t2m, line.strip()))
                text_list.append(pjoin(text_dir_t2m, line.strip()))
        with cs.open(split_file_s2m, "r") as f:
            for line in f.readlines():
                id_list.append(pjoin(motion_dir_s2m, line.strip()))
                text_list.append(pjoin(text_dir_s2m, line.strip()))
        self.id_list = id_list # len 23384
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
            id_list = id_list[:maxdata]
        elif visualize:
            maxdata = 3000
            id_list = id_list[:maxdata]
        
        if progress_bar:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Mixed {split_file_t2m.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(self.id_list)
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            try:
                motion = np.load(name + ".npy")
                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                              200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_list[i] + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(name)
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print('num of data Mixed' , len(self.name_list))

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer
        
    def __getitem__(self, item):
        '''
        HumanML3D: text + motion
        '''
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            length = (length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            length = (length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - length)
        motion = motion[idx:idx + length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        
        
        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")
        
        
        return (
            motion,
            length,
        )
        


class TextStyle2MotionDatasetV1(data.Dataset):

    def __init__(
        self,
        cfg,
        stage,
        phase,
        dataset_name,
        mean,
        std,
        split_file_t2m,
        split_file_s2m,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir_t2m,
        motion_dir_s2m,
        tiny=False,
        debug=False,
        visualize=False,
        progress_bar=True,
        **kwargs,
    ):
        self.stage = stage
        self.phase = phase
        '''
        HumanML3D: text + motion
        '''
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        # Style
        print('dataset_name', dataset_name)
        self.sty_name2num = sty_name2num_dict[dataset_name]
        self.sty_num2name = dict(zip(self.sty_name2num.values(), self.sty_name2num.keys()))
        self.nstyles = len(self.sty_name2num.keys())
        # whether use SRA pred labels for t2m branch
        self.USE_PRED_LABEL = cfg.TRAIN.USE_PRED_LABEL if "USE_PRED_LABEL" in cfg.TRAIN else False
        # load SRA pred labels
        if self.USE_PRED_LABEL:
            sra_pred_dict = {}
            with cs.open('./datasets/humanml3d_sra_stylelabel.txt', "r") as f:
                for line in f.readlines():
                    name, pred_label = line.strip().split('#')
                    sra_pred_dict[name] = pred_label 

        print('dir_t2m', motion_dir_t2m, split_file_t2m)
        print('dir_s2m', motion_dir_s2m, split_file_s2m)
        text_dir_t2m = pjoin(os.path.dirname(motion_dir_t2m), 'texts')
        if self.stage == 'vae':
            text_dir_s2m = pjoin(os.path.dirname(motion_dir_s2m), 'texts')
        else:
            text_dir_s2m = pjoin(os.path.dirname(motion_dir_s2m), 'texts_content')

        data_dict = {}
        id_list = []
        with cs.open(split_file_t2m, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list # len 23384
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
            id_list = id_list[:maxdata]
        elif visualize:
            maxdata = 300
            id_list = id_list[:maxdata]

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file_t2m.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
            
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            try:
                motion = np.load(pjoin(motion_dir_t2m, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                              200):
                    continue
                text_data = []
                if self.USE_PRED_LABEL:
                    label_style = sra_pred_dict[name]
                else:
                    label_style = "Neutral"
                flag = False
                with cs.open(pjoin(text_dir_t2m, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "label_sty": label_style,
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "label_sty": label_style,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(name)
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print('num of data HumanML3D' , len(self.name_list))

        '''
        100STYLE: text + motion_cont + motion_sty
        '''
        data_dict_style = {}
        id_list = []
        with cs.open(split_file_s2m, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
            id_list = id_list[:maxdata]
        
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading 100STYLE {split_file_s2m.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        
        new_name_list = []
        length_list = []
        self.data_in_style_label = {} # 按style类别整理
        for i, name in enumerator:
            motion_sty = np.load(pjoin(motion_dir_s2m, name + ".npy"))
            label_sty, label_cont, _, _ = name.split('_')
            # text
            text_data = []
            with cs.open(pjoin(text_dir_s2m, name + ".txt")) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    text_dict["caption"] = caption
                    text_dict["tokens"] = tokens
            #
            data_dict_style[name] = {
                    "motion": motion_sty,
                    "length": len(motion_sty),
                    "label_sty": label_sty,
                    "label_cont": label_cont,
                    "text": [text_dict],
                    }
            new_name_list.append(name)
            length_list.append(len(motion_sty))
            if label_sty not in self.data_in_style_label:
                self.data_in_style_label[label_sty] = [name]
            else:
                self.data_in_style_label[label_sty].append(name)

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        
        self.name_list_sty = name_list
        self.data_dict_sty = data_dict_style
        self.style_label = self.data_in_style_label.keys()
        if 'Neutral' in self.data_in_style_label:
            self.neutral_list = self.data_in_style_label['Neutral']
        else: # 在没有完整载入数据集的模式可能报错,所以先填一个动作
            self.neutral_list = [name_list[0]]
        print('num of data 100STYLE' , len(self.name_list_sty))

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer
        
    def __getitem__(self, item):
        '''
        HumanML3D: text + motion
        '''
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        t2m_motion, t2m_m_length, text_list = data["motion"], data["length"], data["text"]
        if np.any(np.isnan(t2m_motion)):
            print(self.name_list[idx])
            print(t2m_motion)
            raise ValueError("nan in motion")
        # Randomly select a caption
        text_data = random.choice(text_list)
        t2m_caption, t2m_tokens = text_data["caption"], text_data["tokens"]

        if len(t2m_tokens) < self.max_text_len:
            # pad with "unk"
            t2m_tokens = ["sos/OTHER"] + t2m_tokens + ["eos/OTHER"]
            t2m_sent_len = len(t2m_tokens)
            t2m_tokens = t2m_tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - t2m_sent_len)
        else:
            # crop
            t2m_tokens = t2m_tokens[:self.max_text_len]
            t2m_tokens = ["sos/OTHER"] + t2m_tokens + ["eos/OTHER"]
            t2m_sent_len = len(t2m_tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in t2m_tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        t2m_pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        t2m_word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            t2m_m_length = (t2m_m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            t2m_m_length = (t2m_m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(t2m_motion) - t2m_m_length)
        t2m_motion = t2m_motion[idx:idx + t2m_m_length]
        "Z Normalization"
        t2m_motion = (t2m_motion - self.mean) / self.std
        
        if self.USE_PRED_LABEL:
            t2m_style = self.data_dict_sty[random.choice(self.neutral_list)]["motion"] # 这个变量实际训练用不到，仅占位
            t2m_style = (t2m_style - self.mean) / self.std
            t2m_style_label_name = data["label_sty"]
            t2m_style_label = self.sty_name2num[t2m_style_label_name]
        else:
            # 风格为Neutral
            t2m_style = self.data_dict_sty[random.choice(self.neutral_list)]["motion"]
            t2m_style = (t2m_style - self.mean) / self.std
            t2m_style_label_name = "Neutral"
            t2m_style_label = self.sty_name2num[t2m_style_label_name]

        # debug check nan
        if np.any(np.isnan(t2m_motion)):
            print(self.name_list[idx])
            print(t2m_motion)
            raise ValueError("nan in motion")
        
        '''
        100STYLE: motion_style
        '''
        # print(len(self.data_dict_sty), self.data_dict_sty[self.name_list_sty[0]]["motion_sty"].shape)
        s2m_data = self.data_dict_sty[random.choice(self.name_list_sty)]
        s2m_motion = (s2m_data["motion"] - self.mean) / self.std
        s2m_m_length = s2m_data["length"]
        s2m_style_label_name = s2m_data["label_sty"]
        
        # 选择一个内容不同、风格相同的动作，作为风格参考动作
        motion_same_style = self.data_in_style_label[s2m_style_label_name]
        for loop_i in range(100):
            if loop_i == 99:
                raise RuntimeError("death looping... can't find motion in different content")
            motion_cont_name = random.choice(motion_same_style)
            if self.data_dict_sty[motion_cont_name]["label_cont"] != s2m_data["label_cont"]:
                break
        s2m_style = self.data_dict_sty[motion_cont_name]["motion"]
        s2m_style = (s2m_style - self.mean) / self.std
        s2m_style_label = self.sty_name2num[s2m_style_label_name]
        # print('s2m motion', s2m_style_label_name, s2m_data["label_cont"])
        # print('内容不同、风格相同的动作', self.data_dict_sty[motion_cont_name]["label_sty"], self.data_dict_sty[motion_cont_name]["label_cont"])
        
        # 文本为构造文本 
        # Randomly select a caption
        text_data = random.choice(s2m_data["text"])
        s2m_caption, tokens = text_data["caption"], text_data["tokens"]
        
        # triplet loss
        # pos_name = random.sample(motion_same_style, 1)[0]
        # pos = self.data_dict_sty[pos_name]["motion"]
        pos = s2m_style
        
        for loop_i in range(100):
            if loop_i == 99:
                raise RuntimeError("death looping... can't find motion in different style")
            diff_label_sty = random.sample(self.style_label, 1)[0]
            if diff_label_sty != s2m_data["label_sty"]:
                break
        motion_diff_style = self.data_in_style_label[diff_label_sty]
        neg_name = random.choice(motion_diff_style)
        neg = self.data_dict_sty[neg_name]["motion"]
        # print('neg_name', neg_name, self.data_dict_sty[neg_name]["label_cont"])
        
        return (
            t2m_motion,
            t2m_m_length,
            t2m_caption,
            t2m_word_embeddings,
            t2m_pos_one_hots,
            t2m_sent_len,
            t2m_style,
            t2m_style_label,
            
            s2m_motion,
            s2m_m_length,
            s2m_caption,
            s2m_style,
            s2m_style_label_name,
            s2m_style_label,
            pos,
            neg,
        )


class Text2Motion_Testset1(data.Dataset):
    def __init__(
        self,
        cfg,
        dataset_name,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list
        num_styles = cfg.TESTSET1.NUM_STYLES
        style_dict = np.load(pjoin('./datasets', 'testset1', f'style_dict{num_styles}.npy'), allow_pickle=True).item()
        if num_styles == 100:
             self.sty_name2num = sty_name2num_dict["difusion"]
        elif num_styles == 16:
             self.sty_name2num = sty_name2num_dict["hm_bfa"]
        self.sty_num2name = dict(zip(self.sty_name2num.values(), self.sty_name2num.keys()))
        text_dir_cont = text_dir
        text_dir = text_dir.replace('texts', f'testset1_{num_styles}styles')
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
            id_list = id_list[:maxdata]

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        print('id_list',len(id_list))

        new_name_list = []
        length_list = []
        for i, name in enumerator:
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                            200):
                continue
            k_styles = style_dict[name]
            text_data = []
            text_data_cont = []
            flag = False
            
            for style in k_styles:
                with cs.open(pjoin(text_dir_cont, name + ".txt")) as f_cont:
                        lines_cont = f_cont.readlines()
                   
                with cs.open(pjoin(text_dir, name + f"_{style}.txt")) as f:
                    for i, line in enumerate(f.readlines()):
                        text_dict = {}
                        text_dict_cont = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        # content
                        line_cont = lines_cont[i]
                        line_split = line_cont.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        text_dict_cont["caption"] = caption
                        text_dict_cont["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                            text_data_cont.append(text_dict_cont)
                        else:
                            # print('flag is False!!!!!',f_tag,to_tag,name,line)
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "text_cont": [text_dict_cont],
                                    "style_label": self.sty_name2num[style],
                                    "style_label_name": style,
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "text_cont": text_data_cont,
                        "style_label": self.sty_name2num[style],
                        "style_label_name": style,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        print("num of data", len(self.name_list) - self.pointer)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, text_cont_list, style_label, style_label_name = \
            data["motion"], data["length"], data["text"], data["text_cont"], data["style_label"], data["style_label_name"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        text_data_cont = text_cont_list[text_list.index(text_data)]
        caption, _ = text_data["caption"], text_data["tokens"]
        # tokens是用来计算t2m指标的，所以只计算text_data_cont
        caption_cont, tokens = text_data_cont["caption"], text_data_cont["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                print(token, self.name_list[idx], tokens)
                raise ValueError 
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            print(self.name_list[idx])
            raise ValueError("nan in motion")

        return (
            motion,
            m_length,

            caption,
            
            caption_cont, 
            sent_len,
            word_embeddings,
            pos_one_hots,
            "_".join(tokens),

            style_label,
            style_label_name
        )

class TextStyle2Motion_Testset1(data.Dataset):

    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file_t2m,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir_t2m,
        motion_dir_s2m,
        tiny=False,
        debug=False,
        visualize=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file_t2m, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list
        text_dir_t2m = pjoin(os.path.dirname(motion_dir_t2m), 'texts')
        num_styles = cfg.TESTSET1.NUM_STYLES
        style_dict = np.load(pjoin('./datasets', 'testset1', f'style_dict{num_styles}.npy'), allow_pickle=True).item()
        if num_styles == 100:
             self.sty_name2num = sty_name2num_dict["difusion"]
        elif num_styles == 16:
             self.sty_name2num = sty_name2num_dict["hm_bfa"]
        self.sty_num2name = dict(zip(self.sty_name2num.values(), self.sty_name2num.keys()))
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
            id_list = id_list[:maxdata]

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file_t2m.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        
        print('id_list',len(id_list))

        # load style reference motions
        style_motions = {}
        for style in self.sty_name2num.keys():
            if num_styles == 100:
                style_motions[style] = np.load(pjoin(motion_dir_s2m, f"{style}_FW_1420_1560.npy"))
            elif num_styles == 16:
                style_motions[style] = np.load(pjoin(motion_dir_s2m, f"{style}_01_1020_1160.npy"))

        new_name_list = []
        length_list = []
        for i, name in enumerator:
            motion = np.load(pjoin(motion_dir_t2m, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                            200):
                continue
            k_styles = style_dict[name]
            text_data = []
            text_data_cont = []
            flag = False
            
            for style in k_styles:
                # style = "Neutral" # only text
                with cs.open(pjoin(text_dir_t2m, name + ".txt")) as f:
                    for i, line in enumerate(f.readlines()):
                        text_dict = {}
                        text_dict_cont = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                            text_data_cont.append(text_dict_cont)
                        else:
                            # print('flag is False!!!!!',f_tag,to_tag,name,line)
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "text_cont": [text_dict_cont],
                                    "style_label": self.sty_name2num[style],
                                    "style_label_name": style,
                                    "style_motion": style_motions[style],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "text_cont": text_data_cont,
                        "style_label": self.sty_name2num[style],
                        "style_label_name": style,
                        "style_motion": style_motions[style],
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        print("num of data", len(self.name_list) - self.pointer)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, text_cont_list, style_label, style_label_name, style_motion = \
            data["motion"], data["length"], data["text"], data["text_cont"], data["style_label"], data["style_label_name"], data["style_motion"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                print(token, self.name_list[idx], tokens)
                raise ValueError 
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        style_motion = (style_motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            print(self.name_list[idx])
            raise ValueError("nan in motion")

        return (
            motion,
            m_length,
            caption,
            caption,
            sent_len,
            word_embeddings,
            pos_one_hots,
            "_".join(tokens),

            style_label,
            style_label_name,
            style_motion
        )
        # return caption, motion, m_length
        
sty_name2num_dict = {}
sty_name2num_dict["difusion"] = {
    "Aeroplane": 0,
    "Akimbo": 1,
    "Angry": 2,
    "ArmsAboveHead": 3,
    "ArmsBehindBack": 4,
    "ArmsBySide": 5,
    "ArmsFolded": 6,
    "Balance": 7,
    "BeatChest": 8,
    "BentForward": 9,
    "BentKnees": 10,
    "BigSteps": 11,
    "BouncyLeft": 12,
    "BouncyRight": 13,
    "Cat": 14,
    "Chicken": 15,
    "CrossOver": 16,
    "Crouched": 17,
    "CrowdAvoidance": 18,
    "Depressed": 19,
    "Dinosaur": 20,
    "DragLeftLeg": 21,
    "DragRightLeg": 22,
    "Drunk": 23,
    "DuckFoot": 24,
    "Elated": 25,
    "FairySteps": 26,
    "Flapping": 27,
    "FlickLegs": 28,
    "Followed": 29,
    "GracefulArms": 30,
    "HandsBetweenLegs": 31,
    "HandsInPockets": 32,
    "Heavyset": 33,
    "HighKnees": 34,
    "InTheDark": 35,
    "KarateChop": 36,
    "Kick": 37,
    "LawnMower": 38,
    "LeanBack": 39,
    "LeanLeft": 40,
    "LeanRight": 41,
    "LeftHop": 42,
    "LegsApart": 43,
    "LimpLeft": 44,
    "LimpRight": 45,
    "LookUp": 46,
    "Lunge": 47,
    "March": 48,
    "Monk": 49,
    "Morris": 50,
    "Neutral": 51,
    "Old": 52,
    "OnHeels": 53,
    "OnPhoneLeft": 54,
    "OnPhoneRight": 55,
    "OnToesBentForward": 56,
    "OnToesCrouched": 57,
    "PendulumHands": 58,
    "Penguin": 59,
    "PigeonToed": 60,
    "Proud": 61,
    "Punch": 62,
    "Quail": 63,
    "RaisedLeftArm": 64,
    "RaisedRightArm": 65,
    "RightHop": 66,
    "Roadrunner": 67,
    "Robot": 68,
    "Rocket": 69,
    "Rushed": 70,
    "ShieldedLeft": 71,
    "ShieldedRight": 72,
    "Skip": 73,
    "SlideFeet": 74,
    "SpinAntiClock": 75,
    "SpinClock": 76,
    "Star": 77,
    "StartStop": 78,
    "Stiff": 79,
    "Strutting": 80,
    "Superman": 81,
    "Swat": 82,
    "Sweep": 83,
    "Swimming": 84,
    "SwingArmsRound": 85,
    "SwingShoulders": 86,
    "Teapot": 87,
    "Tiptoe": 88,
    "TogetherStep": 89,
    "TwoFootJump": 90,
    "WalkingStickLeft": 91,
    "WalkingStickRight": 92,
    "Waving": 93,
    "WhirlArms": 94,
    "WideLegs": 95,
    "WiggleHips": 96,
    "WildArms": 97,
    "WildLegs": 98,
    "Zombie": 99,
}
sty_name2num_dict["difusion_act"] = {
    "Akimbo": 1,
    "ArmsAboveHead": 3,
    "ArmsBehindBack": 4,
    "ArmsBySide": 5,
    "ArmsFolded": 6,
    "BeatChest": 8,
    "BentForward": 9,
    "BentKnees": 10,
    "BigSteps": 11,
    "BouncyLeft": 12,
    "BouncyRight": 13,
    "CrossOver": 16,
    # "Crouched": 17,
    # "FairySteps": 26,
    "FlickLegs": 28,
    "Followed": 29,
    "GracefulArms": 30,
    "HandsBetweenLegs": 31,
    "HandsInPockets": 32,
    "HighKnees": 34,
    "KarateChop": 36,
    "Kick": 37,
    "LeanBack": 39,
    "LeanLeft": 40,
    "LeanRight": 41,
    "LeftHop": 42,
    "LegsApart": 43,
    "LimpLeft": 44,
    "LimpRight": 45,
    "LookUp": 46,
    "Lunge": 47,
    "March": 48,
    # "Neutral": 51,
    # "PendulumHands": 58,
    # "PigeonToed": 60,
    "Punch": 62,
    "RaisedLeftArm": 64,
    "RaisedRightArm": 65,
    "RightHop": 66,
    "Skip": 73,
    "SlideFeet": 74,
    "SpinAntiClock": 75,
    "SpinClock": 76,
    "StartStop": 78,
    "Strutting": 80,
    "Sweep": 83,
    "Teapot": 87,
    "Tiptoe": 88,
    "TogetherStep": 89,
    "TwoFootJump": 90,
    "WalkingStickLeft": 91,
    "WalkingStickRight": 92,
    "Waving": 93,
    "WhirlArms": 94,
    "WideLegs": 95,
    "WiggleHips": 96,
    "WildArms": 97,
    "WildLegs": 98,
}

# 计算 "most" 中不在 "most_act" 中的项
remaining_items = set(sty_name2num_dict["difusion"].items()) - set(sty_name2num_dict["difusion_act"].items())
# 将剩余项重新索引
sty_name2num_dict["difusion_47"] = {k: i for i, (k, v) in enumerate(sorted(remaining_items))}

sty_name2num_dict["style100"] = sty_name2num_dict["difusion"]

sty_name2num_dict["hm_bfa"] = {
    "Angry": 0, "Depressed": 1, "Drunk": 2, "FemaleModel": 3, 
    "Happy": 4, "Heavy": 5, "Hurried": 6, "Lazy": 7,
    "Neutral": 8, "Old": 9, "Proud": 10, "Robot": 11,
    "Sneaky": 12, "Soldier": 13, "Strutting": 14, "Zombie": 15,
}

sty_name2num_dict["bfa"] = sty_name2num_dict["hm_bfa"]