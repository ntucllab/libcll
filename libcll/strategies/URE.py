# Reference: https://github.com/takashiishida/comp
# Ishida, Takashi, et al. "Complementary-label learning for arbitrary losses and models." International Conference on Machine Learning. PMLR, 2019.
import torch
import torch.nn.functional as F
import numpy as np
from libcll.strategies.Strategy import Strategy


class URE(Strategy):
    def __init__(self, **args):
        if args["type"] == "GA" or args["type"] == "NN":
            Q = (
                torch.ones(args["num_classes"], args["num_classes"])
                * 1
                / (args["num_classes"] - 1)
            )
            for k in range(args["num_classes"]):
                Q[k, k] = 0
            args["Q"] = Q
        super().__init__(**args)
        self.beta = 0

    def custom_non_negative_loss(self, output, labels, beta):
        class_priors = self.class_priors.requires_grad_().to(output.device)
        neglog = -F.log_softmax(output, dim=1)
        l = labels.long()
        torch.use_deterministic_algorithms(False)
        counts = torch.bincount(l, minlength=self.num_classes).view(-1, 1)
        torch.use_deterministic_algorithms(True)
        lh = F.one_hot(l, self.num_classes).float()
        neg_vector = torch.matmul(lh.t(), neglog)
        loss_vector = (self.Qinv * neg_vector).sum(dim=1) * class_priors
        vc = (1 / counts).nan_to_num(0).view(-1)
        loss_vector = loss_vector * vc
        loss = loss_vector[loss_vector > -beta].sum()
        loss.requires_grad_()
        loss_vector.requires_grad_()
        return loss, loss_vector

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        if self.type == "TGA" or self.type == "GA":
            loss, loss_vector = self.custom_non_negative_loss(
                out,
                y,
                torch.inf,
            )
            if torch.min(loss_vector) < 0:
                loss = -loss_vector[loss_vector < 0].sum()
            self.log("Train_Loss", loss)
            return loss

        elif self.type == "TNN" or self.type == "NN":
            loss, loss_vector = self.custom_non_negative_loss(
                out,
                y,
                self.beta,
            )
            self.log("Train_Loss", loss)
            return loss
        else:
            raise NotImplementedError(
                'The type of URE must be chosen from "NN", "TNN", "GA" or "TGA".'
            )
