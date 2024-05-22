# Wei-I Lin and Hsuan-Tien Lin. "Reduction from Complementary-Label Learning to Probability Estimates" in PAKDD. 2023.
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from libcll.strategies.Strategy import Strategy


class CPE(Strategy):
    def __init__(self, **args):
        super().__init__(**args)
        if self.type == "T":
            self.Q = self.Q.log()
            self.model.register_parameter(
                name="Q", param=torch.nn.Parameter(self.Q, requires_grad=True)
            )

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        out = F.softmax(out, dim=1)
        if self.type == "I":
            loss = F.nll_loss(out.log(), y.long())
        elif self.type == "F":
            Q = self.Q
            out = torch.mm(out, Q) + 1e-6
            loss = F.nll_loss(out.log(), y.long())
        elif self.type == "T":
            Q = self.model.Q
            Q = F.softmax(Q, dim=1)
            out = torch.mm(out, Q) + 1e-6
            loss = F.nll_loss(out.log(), y.long())
        else:
            raise NotImplementedError(
                'The type of CPE must be chosen from "I", "F" or "T".'
            )
        self.log("Train_Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        if self.valid_type != "SCEL" and self.valid_type != "Accuracy":
            raise ValueError(
                f"The Validation Loss of CPE can only be SCEL or Accuracy."
            )
        out = F.softmax(out, dim=1)
        if self.type == "I":
            Q = self.Q
        elif self.type == "F":
            Q = self.Q
            out = torch.mm(out, Q) + 1e-6
        elif self.type == "T":
            Q = self.model.Q
            Q = F.softmax(Q, dim=1)
            out = torch.mm(out, Q) + 1e-6
        else:
            raise NotImplementedError(
                'The type of CPE must be chosen from "I", "F" or "T".'
            )

        if self.valid_type == "SCEL":
            val_loss = F.nll_loss(out.log(), y.long())
        if self.valid_type == "Accuracy":
            y_pred = torch.argmin(torch.abs(out.unsqueeze(dim=1) - Q).sum(dim=2), dim=1)
            val_loss = (y_pred == y).sum() / y_pred.shape[0]
        self.val_loss.append(val_loss)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        out = F.softmax(out, dim=1)
        if self.type == "I":
            Q = self.Q
        elif self.type == "F":
            Q = self.Q
            out = torch.mm(out, Q) + 1e-6
        elif self.type == "T":
            Q = self.model.Q
            Q = F.softmax(Q, dim=1)
            out = torch.mm(out, Q) + 1e-6

        y_pred = torch.argmin(torch.abs(out.unsqueeze(dim=1) - Q).sum(dim=2), dim=1)
        acc = (y_pred == y).sum() / y_pred.shape[0]
        self.test_acc.append(acc)
        return {"test_acc": acc}
