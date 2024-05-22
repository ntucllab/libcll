# Yi Gao, Min-Ling Zhang. Discriminative Complementary-Label Learning with Weighted Loss
import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class DM(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        p = 1 - F.softmax(out, dim=1)
        q = F.softmax(p, dim=1) + 1e-6
        w = torch.mul(p / (out.shape[1] - 1), q.log())
        loss = F.nll_loss(q.log(), y.long()) + F.nll_loss(w, y.long())
        self.log("Train_Loss", loss)
        return loss
