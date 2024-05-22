# Reference: https://github.com/takashiishida/comp
# Yu, Xiyu, et al. "Learning with biased complementary labels." Proceedings of the European conference on computer vision (ECCV). 2018.
import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class FWD(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        p = torch.mm(F.softmax(out, dim=1), self.Q) + 1e-6
        loss = F.nll_loss(p.log(), y.long())
        self.log("Train_Loss", loss)
        return loss
