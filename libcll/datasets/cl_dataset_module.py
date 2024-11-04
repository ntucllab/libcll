import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Sampler, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import copy
from .utils import get_transition_matrix, collate_fn_multi_label, collate_fn_one_hot

class IndexSampler(Sampler):
    def __init__(self, index):
        self.index = index

    def __iter__(self):
        ind = torch.randperm(len(self.index))
        return iter(self.index[ind].tolist())

    def __len__(self):
        return len(self.index)

class CLDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_name, 
        dataset_class,
        batch_size=256,
        valid_split=0.1,
        valid_type="URE",
        one_hot=False,
        transition_matrix=None,
        num_cl=1,
        augment=False,
        noise=0.1,
        seed=1126,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.valid_type = valid_type
        self.one_hot = one_hot
        self.transition_matrix = transition_matrix
        self.num_cl = num_cl
        self.augment = augment
        self.noise = noise
        self.seed = seed
    
    def setup(self, stage=None):
        pl.seed_everything(self.seed, workers=True)
        self.train_set = self.dataset_class.build_dataset(self.dataset_name, train=True, num_cl=self.num_cl, transition_matrix=self.transition_matrix, noise=self.noise, seed=self.seed)
        self.test_set = self.dataset_class.build_dataset(self.dataset_name, train=False)
        idx = np.arange(len(self.train_set))
        np.random.shuffle(idx)
        self.train_idx = idx[: int(len(self.train_set) * (1 - self.valid_split))]
        self.valid_idx = idx[int(len(self.train_set) * (1 - self.valid_split)) :]
        if self.valid_type == "Accuracy":
            for i in self.valid_idx:
                self.train_set.targets[i] = self.train_set.true_targets[i].view(1)
    
    def train_dataloader(self):
        train_sampler = IndexSampler(self.train_idx)
        train_loader = DataLoader(
            self.train_set,
            sampler=train_sampler,
            batch_size=self.batch_size,
            collate_fn=(
                collate_fn_multi_label
                if not self.one_hot
                else lambda batch: collate_fn_one_hot(
                    batch, num_classes=self.train_set.num_classes
                )
            ),
            shuffle=False,
            num_workers=4, 
            # persistent_workers=True, 
            # pin_memory=True, 
        )
        return train_loader

    def val_dataloader(self):
        if self.valid_split:
            valid_sampler = IndexSampler(self.valid_idx)
            valid_loader = DataLoader(
                self.train_set,
                sampler=valid_sampler,
                batch_size=self.batch_size,
                collate_fn=collate_fn_multi_label,
                shuffle=False,
                num_workers=4, 
                # persistent_workers=True, 
                # pin_memory=True, 
            )
        else:
            valid_loader = DataLoader(
                self.test_set, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4, 
                # persistent_workers=True, 
                # pin_memory=True, 
            )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            # persistent_workers=True, 
            # pin_memory=True, 
        )
        return test_loader
    
    def get_distribution_info(self):
        Q = torch.zeros((self.train_set.num_classes, self.train_set.num_classes))
        for idx in self.train_idx:
            Q[self.train_set.true_targets[idx].long()] += torch.histc(
                self.train_set.targets[idx].float(), self.train_set.num_classes, 0, self.train_set.num_classes
            )
        class_priors = Q.sum(dim=0)
        Q = Q / Q.sum(dim=1).view(-1, 1)
        if self.transition_matrix == "noisy":
            Q = get_transition_matrix("strong", self.train_set.num_classes, self.seed)
        return (
            Q,
            class_priors,
        )