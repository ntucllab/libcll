import torch
import torchvision
from libcll.datasets.cl_base_dataset import CLBaseDataset
import numpy as np
from PIL import Image
import urllib.request
from tqdm import tqdm
import pickle
import gdown
import os


class CLTiny_ImageNet10(torchvision.datasets.ImageFolder, CLBaseDataset):
    """

    Real-world complementary-label dataset. Call ``gen_complementary_target()`` if you want to access synthetic complementary labels.

    Parameters
    ----------
    root : str
        path to store dataset file.

    train : bool
        training set if True, else testing set.

    transform : callable, optional
        a function/transform that takes in a PIL image and returns a transformed version.

    target_transform : callable, optional
        a function/transform that takes in the target and transforms it.

    download : bool
        if true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    num_cl : int
        the number of real-world complementary labels of each data chosen from [1, 3].

    Attributes
    ----------
    data : Tensor
        the feature of sample set.

    targets : Tensor
        the complementary labels for corresponding sample.

    true_targets : Tensor
        the ground-truth labels for corresponding sample.

    num_classes : int
        the number of classes.

    input_dim : int
        the feature space after data compressed into a 1D dimension.

    """

    def __init__(
        self,
        root="./data/imagenet10",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        num_cl=1,
    ):
        if train:
            super(CLTiny_ImageNet10, self).__init__(
                root=os.path.join(root, "train"),
                transform=transform,
                target_transform=target_transform,
            )
            label_to_folder = {}
            file_to_cl = {}
            with open(os.path.join(root, "words.txt"), "r") as f:
                lines = [line.strip() for line in f.readlines()]
                for line in lines:
                    folder, labels = line.split("\t")
                    label = labels.split(",")[0]
                    label_to_folder[label] = folder
            with open(os.path.join(root, "cll_human_words_tiny_10.txt"), "r") as f:
                lines = [line.strip() for line in f.readlines()]
                for line in lines:
                    file_name, labels = line.split("\t")
                    file_name = os.path.basename(file_name)
                    labels = [
                        self.class_to_idx[label_to_folder[label[1:-1]]]
                        for label in labels.split(", ")
                    ][:num_cl]
                    file_to_cl[file_name] = labels
            self.true_targets = torch.Tensor(self.targets)
            self.targets = [
                torch.Tensor(file_to_cl[os.path.basename(self.samples[i][0])])
                for i in range(len(self.samples))
            ]
        else:
            super(CLTiny_ImageNet10, self).__init__(
                root=os.path.join(root, "val"),
                transform=transform,
                target_transform=target_transform,
            )
        self.num_classes = 10
        self.input_dim = 3 * 64 * 64

    def __getitem__(self, index):
        path, target = self.samples[index][0], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
