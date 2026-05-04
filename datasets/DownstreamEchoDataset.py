from omegaconf import DictConfig

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

from load_video import loadvideo_decord, loadvideo_decord_echojepa
from EchoAugmentations import EchoAugmentations, PanEchoAugmentations, EchoJEPAAugmentations
from echoprime import ECHOPRIME_MEAN, ECHOPRIME_STD, crop_and_scale

class DownstreamEchoDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        mode: str,
        target: str,
        first_n_samples: int = None
    ):
        super(DownstreamEchoDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.target = target
        self.first_n_samples = first_n_samples
        self.task_type = cfg.downstream_task_echo.task_type

        self.clip_len = cfg.downstream_task_echo.num_frames
        self.sampling_rate = self.cfg.downstream_task_echo.sampling_rate

        filelist = pd.read_csv(cfg.downstream_task_echo.paths[f'data_{self.mode}'])
        self.paths = list(filelist['filename'])

        if self.task_type == 'segmentation':
            self.paths_seg = list(filelist['filename_seg'])
        else:
            self.labels = torch.load(cfg.downstream_task_echo.paths[f'labels_{self.mode}'])

        if self.first_n_samples:
            self.paths = self.paths[0:self.first_n_samples]
            if self.task_type == 'segmentation':
                self.paths_seg = self.paths_seg[0:self.first_n_samples]
            else:
                self.labels = self.labels[0:self.first_n_samples]

        if self.task_type == 'regression':
            if 'target_list' in cfg.downstream_task_echo:
                self.lower_bound = cfg.downstream_task_echo.target_list[target].lower_bound
                self.upper_bound = cfg.downstream_task_echo.target_list[target].upper_bound
                self.num_classes = self.upper_bound - self.lower_bound
            else:
                self.num_classes = len(self.labels[0])
        elif self.task_type in ['binary_classification', 'multiclass_classification', 'multilabel_classification']:
            self.num_classes = len(self.labels[0])
        
        self._init_augmentations()
    
    def _init_augmentations(self):
        if self.cfg.model.echo.model_name == 'echoprime':
            self.echo_transform_and_augment_fn = EchoAugmentations(self.cfg.downstream_task_echo, crop_size=self.cfg.model.echo.img_size, task_type='pretrain')
        elif self.cfg.model.echo.model_name == 'panecho':
            self.echo_transform_and_augment_fn = PanEchoAugmentations(self.cfg.downstream_task_echo, crop_size=self.cfg.model.echo.img_size, task_type='pretrain')
        elif self.cfg.model.echo.model_name == 'echojepa':
            self.echo_transform_and_augment_fn = EchoJEPAAugmentations(self.cfg.downstream_task_echo, crop_size=self.cfg.model.echo.img_size, task_type='pretrain')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # load echo
        if self.cfg.model.echo.model_name == 'echoprime' or self.cfg.model.echo.model_name == 'mvit_v2_s':
            echo = self._load_echoprime_echo(idx)
    
        elif self.cfg.model.echo.model_name == 'panecho':
            echo = self._load_panecho_echo(idx)
        
        elif self.cfg.model.echo.model_name == 'echojepa':
            echo = self._load_echojepa_echo(idx)
            
        if self.task_type == 'regression':
            if hasattr(self, 'lower_bound'):
                label = self.labels[idx][..., self.lower_bound:self.upper_bound].squeeze() # (nb_classes,)
            else:
                label = self.labels[idx].squeeze()
        elif self.task_type == 'binary_classification':
            label = self.labels[idx].argmax(dim=-1) # e.g. tensor(0)
        elif self.task_type in ['multiclass_classification', 'multilabel_classification']:
            label = self.labels[idx]

        return {
            'echo': echo,
            'label': label,
            'filename': os.path.basename(path)
        }

    def _load_echoprime_echo(self, idx):

        echo_path = self.paths[idx]

        out = loadvideo_decord(echo_path, 'val', self.cfg.dataset.echo.sampling_rate, self.cfg.model.echo.num_frames, repeat_or_pad='pad')
        echo = out['video']
        
        echo = np.array(echo)
        echo_augm = np.zeros((len(echo), self.cfg.model.echo.img_size, self.cfg.model.echo.img_size, self.cfg.model.echo.num_channels))
        for i in range(len(echo_augm)):
            echo_augm[i] = crop_and_scale(echo[i])

        echo_augm = torch.from_numpy(echo_augm).float()
        echo_augm = echo_augm.permute(3, 0, 1, 2)
        echo_augm.sub_(ECHOPRIME_MEAN).div_(ECHOPRIME_STD)
        
        return echo_augm
    
    def _load_panecho_echo(self, idx):

        echo_path = self.paths[idx]
        
        out = loadvideo_decord(echo_path, 'val', self.cfg.dataset.echo.sampling_rate, self.cfg.model.echo.num_frames, repeat_or_pad='repeat')
        echo = out['video'] # (T, H, W, C)
        echo = echo[:, :, :, [2, 1, 0]] # convert to BGR to make it the same as the original code
        
        echo = torch.from_numpy(echo)
        echo = echo.permute(0, 3, 1, 2) # (T, C, H, W)
        echo_augm = self.echo_transform_and_augment_fn(echo, use_augmentations=self.mode == 'train')
        echo_augm = echo_augm.permute(1, 0, 2, 3)
        
        return echo_augm
    
    def _load_echojepa_echo(self, idx):

        echo_path = self.paths[idx]

        echo = loadvideo_decord_echojepa(
            echo_path,
            fpc=self.cfg.model.echo.num_frames,
            frame_step=None,
            duration=None,
            fps=self.cfg.dataset.echo.fps
        )['video'] # (T, H, W, C)

        echo_augm = self.echo_transform_and_augment_fn(echo, use_augmentations=self.mode == 'train') # (C, T, H, W)
        
        return echo_augm