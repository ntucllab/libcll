# Feng, L., Kaneko, T., Han, B., Niu, G., An, B., and Sugiyama, M. "Learning with multiple complementary labels."" In ICML, 2020.
import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class MCL(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        p = F.softmax(out, dim=1)
        p = ((1 - y) * p).sum(dim=1)
        if self.type == "MAE":
            loss = torch.ones(y.shape[0], device=x.device) - p
        elif self.type == "EXP":
            loss = torch.exp(-p)
        elif self.type == "LOG":
            loss = -torch.log(p)
        else:
            raise NotImplementedError(
                'The type of MCL must be chosen from "MAE", "EXP" or "LOG".'
            )
        loss = ((2 * self.num_classes - 2) * loss / y.sum(dim=1)).sum()
        self.log("Train_Loss", loss)
        return loss
