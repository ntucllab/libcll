import torch
import numpy as np
import inspect


def collate_fn_multi_label(data):
    """

    Collate function. Duplicate image input to the length of corresponding complementary labels if the data point has multiple complementary labels.

    Parameters
    ----------
    data:
        list of (image input, target) from dataset.

    Returns
    -------
    (Tensor of **image inputs**, Tensor of **targets**)
    """
    x = []
    y = []
    for x_i, y_i in data:
        x.append(
            x_i.unsqueeze(0).repeat(y_i.shape[0], *[1 for _ in range(len(x_i.shape))])
        )
        y.append(y_i)
    return torch.cat(x, 0), torch.cat(y, 0)


def collate_fn_one_hot(data, num_classes):
    """

    Collate function. Store complementary labels into one-hot vectors.

    Parameters
    ----------
    data:
        list of (image input, target) from dataset.
    num_classes:
        the number of classes.

    Returns
    -------
    (Tensor of **image inputs**, Tensor of **targets**)
    """
    x = []
    y = torch.zeros(len(data), num_classes)
    for i, (x_i, y_i) in enumerate(data):
        x.append(x_i)
        y[i][y_i.long()] = 1
    return torch.stack(x, 0), y


def Uniform(num_classes):
    Q = torch.ones(num_classes, num_classes) - torch.eye(num_classes)
    Q = Q / (num_classes - 1)
    return Q


def Weak(num_classes, seed=1126):
    if num_classes != 10:
        raise ValueError(
            f"Weak Distribution have only implement for num_classes=10 datasets."
        )
    rng = np.random.default_rng(seed=seed)
    Q = torch.zeros((num_classes, num_classes))
    distribution = [0.45 / 3, 0.30 / 3, 0.25 / 3] * 3
    for i in range(num_classes):
        rng.shuffle(distribution)
        Q[i] = torch.tensor(distribution[:i] + [0] + distribution[i:])
    return Q


def Strong(num_classes, seed=1126):
    if num_classes != 10:
        raise ValueError(
            f"Strong Distribution have only implement for num_classes=10 datasets."
        )
    rng = np.random.default_rng(seed=seed)
    Q = torch.zeros((num_classes, num_classes))
    distribution = [0.75 / 3, 0.24 / 3, 0.01 / 3] * 3
    for i in range(num_classes):
        rng.shuffle(distribution)
        Q[i] = torch.tensor(distribution[:i] + [0] + distribution[i:])
    return Q


def biased_on_one(num_classes, max_prob=0.9, seed=1126):
    rng = np.random.default_rng(seed=seed)
    Q = (
        (torch.ones(num_classes, num_classes) - torch.eye(num_classes))
        * (1 - max_prob)
        / (num_classes - 2)
    )
    for i in range(num_classes):
        biased_class = (i + rng.randint(1, num_classes)) % num_classes
        Q[i, biased_class] = max_prob
    return Q


def partial_uniform(num_classes, partial=6):
    Q = (torch.ones(num_classes, num_classes) - torch.eye(num_classes)) * 0.001
    for i in range(num_classes):
        p = torch.ones(num_classes)
        p[i] = 0
        idx = torch.multinomial(p, partial)
        Q[i, idx] = (1.0 - 0.001 * (num_classes - partial - 1)) / partial
    return Q


def noisy(num_classes, noise=0.1, seed=1126):
    Q = Strong(num_classes, seed)
    Q = (1 - noise) * Q + noise * torch.ones(num_classes, num_classes) / num_classes
    return Q


Q_LIST = {
    "uniform": Uniform,
    "weak": Weak,
    "strong": Strong,
    "noisy": noisy,
}


def get_transition_matrix(transition_matrix, num_classes, noise=0.1, seed=1126):
    """

    Return desired class transition probability matrix.

    Parameters
    ----------
    transition_matrix : str
        the name of transition matrix chosen from "uniform", "weak", "strong", and "noise"

    num_classes : int
        the number of classes.

    noise : float
        the noise weight in noisy distribution.

    seed : int
        the random seed.

    Returns
    -------
    a tensor with shape (``num_classes``, ``num_classes``).
    """
    if transition_matrix not in Q_LIST:
        raise ValueError(
            f"The transition matrix {transition_matrix} didn't implemented."
        )
    args = {
        "num_classes": num_classes,
        "noise": noise,
        "seed": seed,
    }
    Q = Q_LIST[transition_matrix]
    Q_args = inspect.getargspec(Q).args
    return Q(**{arg: args[arg] for arg in args if arg in Q_args})
