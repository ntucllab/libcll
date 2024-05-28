# libcll: Complementary Label Learning Benchmark

[![Documentation Status](https://readthedocs.org/projects/libcll/badge/?version=latest)](https://libcll.readthedocs.io/en/latest/?badge=latest) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`libcll` is a Python package designed to make complementary label learning easier for real-world researchers. The package not only implements most of the popular complementary label learning strategies, including CPE, a SOTA algorithm in 2023 but also features CLCIFAR10 and CLCIFAR20 datasets, both of which collect complementary labels from humans.  In addition, the package provides a unified interface for adding more strategies, datasets, and models. 

# Installation

- Python version >= 3.8, <= 3.10
- Pytorch version >= 1.11, <= 2.0
- Pytorch Lightning version >= 2.0
- To install `libcll` and develop locally:

```
cd libcll
pip install -e .
```

# Running

## Supported Strategies

| Strategies                                                | Type             | Description                                                  |
| --------------------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| [SCL](https://arxiv.org/pdf/2007.02235.pdf)               | NL, EXP          | Surrogate Complementary Loss with the negative log loss (NL) or with the exponential loss (EXP) |
| [URE](https://arxiv.org/pdf/1810.04327.pdf)               | NN, GA, TNN, TGA | Unbiased Risk Estimator whether with gradient ascent (GA) or empirical transition matrix (T) |
| [FWD](https://arxiv.org/pdf/1711.09535.pdf)               | None             | Forward Correction                                           |
| [DM](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf) | None             | Discriminative Models with Weighted Loss                     |
| [CPE](https://arxiv.org/pdf/2209.09500.pdf)               | I, F, T          | Complementary Probability Estimates with different transition matrices (I, F, T) |
| [MCL](https://arxiv.org/pdf/1912.12927.pdf)               | MAE, EXP, LOG    | Multiple Complementary Label learning with different errors (MAE, EXP, LOG) |

## Supported Datasets

| Dataset     | Number of Classes | Input Size  | Description                                                  |
| ----------- | --------------- | ----------- | ------------------------------------------------------------ |
| MNIST       | 10              | 28 x 28     | Grayscale images of handwritten digits (0 to 9).             |
| FMNIST      | 10              | 28 x 28     | Grayscale images of fashion items.                           |
| KMNIST      | 10              | 28 x 28     | Grayscale images of cursive Japanese (“Kuzushiji”) characters. |
| Yeast       | 10              | 8           | Features of different localization sites of protein.         |
| Texture     | 11              | 40          | Features of different textures.                              |
| Dermatology | 6               | 130         | Clinical Attributes of different diseases.                              |
| Control     | 6               | 60          | Features of synthetically generated control charts.          |
| Micro ImageNet10   | 10                | 3 x 64 x 64 | Contains images of 10 classes designed for computer vision research. |
| Micro ImageNet20 | 20 | 3 x 64 x 64 | Contains images of 20 classes designed for computer vision research. |
| CIFAR10 | 10 | 3 x 32 x 32 | Colored images of different objects. |
| CIFAR20     | 20              | 3 x 32 x 32 | Colored images of different objects. |
| CLMicro ImageNet10 | 10 | 3 x 64 x 64 | Contains images of 10 classes designed for computer vision research paired with complementary labels annotated by humans. |
| CLMicro ImageNet20 | 20 | 3 x 64 x 64 | Contains images of 20 classes designed for computer vision research paired with complementary labels annotated by humans. |
| CLCIFAR10   | 10              | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by humans. |
| CLCIFAR20   | 20              | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by humans. |

## Quick Start: Complementary Label Learning on MNIST

To reproduce training results with SCL-NL method on MNIST

```shell
python script/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model MLP \
  --dataset MNIST \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
```

or

```
./script/train.sh SCL NL MLP mnist Accuracy
```

# Documentation

Find full [tutorials](https://libcll.readthedocs.io/en/latest/) for libcll.

# Acknowledgment

We thank the following repos for the code sharing.
* [URE and FWD implementation](https://github.com/takashiishida/comp)
* [DM official implementation](http://palm.seu.edu.cn/zhangml/Resources.htm#icml21b)
* [Code structure](https://github.com/ntucllab/imbalanced-DL)
