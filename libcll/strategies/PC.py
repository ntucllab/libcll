import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class PC(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        out = out + F.nll_loss(out, y, reduction='none').view(-1, 1)
        loss = torch.sigmoid(-1 * out).sum(dim=1).mean() - 0.5
        self.log("Train_Loss", loss)
        return loss
