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


class CLCIFAR10(torchvision.datasets.CIFAR10, CLBaseDataset):
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
        root="./data/cifar10",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        num_cl=1,
    ):
        if train:
            dataset_path = f"{root}/clcifar10.pkl"
            if download and not os.path.exists(dataset_path):
                os.makedirs(root, exist_ok=True)
                gdown.download(
                    id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path
                )
            with open(dataset_path, "rb") as f:
                data = pickle.load(f)
            self.data = data["images"]
            self.true_targets = torch.Tensor(data["ord_labels"]).view(-1)
            self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl]
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(CLCIFAR10, self).__init__(
                root, train, transform, target_transform, download
            )
            self.targets = torch.Tensor(self.targets)
        self.num_classes = 10
        self.input_dim = 3 * 32 * 32
