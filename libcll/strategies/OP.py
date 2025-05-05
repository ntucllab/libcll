import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class OP(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        Q_1 = F.softmax(out, 1) + 1e-18
        Q_2 = F.softmax(-out, 1) + 1e-18
        w_ = torch.div(1, Q_2)

        w_ = w_ + 1
        w = F.softmax(w_, 1)

        w = torch.mul(Q_1, w) + 1e-6
        w_1 = torch.mul(w, Q_2.log())
        loss = F.nll_loss(w_1, y.long())
        self.log("Train_Loss", loss)
        return loss
