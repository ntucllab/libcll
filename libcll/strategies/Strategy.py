import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Strategy(pl.LightningModule):
    """
    libcll strategy object

    Parameters
    ----------
    model : nn.Module
        the classification model.

    num_classes : int
        the number of classes.

    valid_type : str
        the type of validation metric.

    type : str
        the type of loss function.

    lr : float
        the learning rate of optimizer.

    Q : Tensor
        the class transition probability matrix. Initialized as uniform distribution if None.

    class_priors : Tensor
        the class priors.

    """

    def __init__(
        self,
        model=None,
        num_classes=10,
        valid_type="SCEL",
        type="NL",
        lr=1e-4,
        Q=None,
        class_priors=None,
    ):
        super().__init__()
        self.model = model
        self.valid_type = valid_type
        self.type = type
        self.lr = lr
        self.num_classes = num_classes
        self.Q = Q
        self.class_priors = class_priors
        self.val_loss = []
        self.test_acc = []
        if self.Q is None:
            self.Q = torch.ones(num_classes, num_classes) * 1 / (num_classes - 1)
            for k in range(num_classes):
                self.Q[k, k] = 0
        self.Q = self.Q.cuda()
        if torch.det(self.Q) != 0:
            self.Qinv = torch.inverse(self.Q)
        else:
            self.Qinv = torch.pinverse(self.Q)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pass

    def compute_ure(self, out, y):
        """
        Compute the Unbiased Risk Estimator loss.
        """
        out = -F.log_softmax(out, dim=1)
        loss_mat = torch.mm(out, self.Qinv.t())
        loss = -F.nll_loss(loss_mat, y.long())
        return loss

    def compute_scel(self, out, y):
        """
        Compute the Surrogate Complementary Esimation Loss.
        """
        out = out.softmax(dim=1)
        out = torch.mm(out, self.Q)
        out = (out + 1e-6).log()
        loss = F.nll_loss(out, y.long())
        return loss

    def compute_acc(self, out, y):
        """
        Compute the Accuracy.
        """
        y_pred = torch.argmax(out, dim=1)
        acc = (y_pred == y).sum() / y_pred.shape[0]
        return acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        if self.valid_type == "URE":
            val_loss = self.compute_ure(out, y)
        elif self.valid_type == "SCEL":
            val_loss = self.compute_scel(out, y)
        elif self.valid_type == "Accuracy":
            val_loss = self.compute_acc(out, y)
        else:
            raise NotImplementedError(
                'The type of validation score must be chosen from "URE", "SCEL" or "Accuracy".'
            )
        self.val_loss.append(val_loss)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_loss).mean()
        self.log(f"Valid_{self.valid_type}", avg_val_loss)
        self.val_loss.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        y_pred = torch.argmax(out, dim=1)
        acc = (y_pred == y).sum() / y_pred.shape[0]
        self.test_acc.append(acc)
        return {"test_acc": acc}

    def on_test_epoch_end(self):
        avg_test_acc = torch.stack(self.test_acc).mean()
        self.log("Test_Accuracy", avg_test_acc)
        self.test_acc.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer
