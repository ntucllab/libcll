import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import gdown
import os
from libcll.datasets.cl_base_dataset import CLBaseDataset
from libcll.datasets.utils import get_transition_matrix

class ACLMicro_ImageNet20(torchvision.datasets.CIFAR10, CLBaseDataset):
    """

    Real-world complementary-label dataset annotated by VLM. Call ``gen_complementary_target()`` if you want to access synthetic complementary labels.

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
        root="./data/imagenet20",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        num_cl=1,
    ):
        if train:
            dataset_path = f"{root}/aclmicro_imagenet20_train.pkl"
            gid = "1a8bD0Nu__eX1qq1ncuOhjsn46GY9wd5X"
        else:
            dataset_path = f"{root}/clmicro_imagenet20_test.pkl"
            gid = "1EdBCrifSrIIUg1ioPWA-ZLEHO53P4NPl"
        if download and not os.path.exists(dataset_path):
            os.makedirs(root, exist_ok=True)
            gdown.download(id=gid, output=dataset_path)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        if train:
            self.true_targets = torch.Tensor(data["ord_labels"]).view(-1)
            self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl]
        else:
            self.targets = torch.Tensor(data["ord_labels"]).view(-1)
        self.data = data["images"]
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 20
        self.input_dim = 3 * 64 * 64
    
    @classmethod
    def build_dataset(self, dataset_name=None, train=True, num_cl=0, transition_matrix=None, noise=None, seed=1126):
        if train:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            dataset = self(
                train=True,
                transform=train_transform,
            )
            if dataset_name == "micro_imagenet20":
                Q = get_transition_matrix(transition_matrix, dataset.num_classes, noise, seed)
                dataset.gen_complementary_target(num_cl, Q)
        else:
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            dataset = self(
                train=False,
                transform=test_transform,
            )
        return dataset
