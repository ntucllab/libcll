import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class SCARCE(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        
        comple_label_mat = torch.zeros_like(out)
        comple_label_mat[torch.arange(comple_label_mat.shape[0]), y.long()] = 1
        comple_label_mat = comple_label_mat.to(out.device)
        pos_loss = -F.logsigmoid(out)
        neg_loss = -F.logsigmoid(-out)
        neg_data_mat = comple_label_mat.float()
        unlabel_data_mat = torch.ones_like(neg_data_mat)
        # calculate negative label loss of negative data
        neg_loss_neg_data_mat = neg_loss * neg_data_mat
        tmp1 = neg_data_mat.sum(dim=0)
        tmp1[tmp1 == 0.] = 1.
        neg_loss_neg_data_vec = neg_loss_neg_data_mat.sum(dim=0) / tmp1
        # calculate positive label loss of unlabeled data
        pos_loss_unlabel_data_mat = pos_loss * unlabel_data_mat
        tmp2 = unlabel_data_mat.sum(dim=0)
        tmp2[tmp2 == 0.] = 1.
        pos_loss_unlabel_data_vec = pos_loss_unlabel_data_mat.sum(dim=0) / tmp2
        # calculate positive label loss of negative data
        pos_loss_neg_data_mat = pos_loss * neg_data_mat
        pos_loss_neg_data_vec = pos_loss_neg_data_mat.sum(dim=0) / tmp1
        # calculate final loss
        prior_vec = 1. / out.shape[1] * torch.ones(out.shape[1])
        prior_vec = prior_vec.to(out.device)
        ccp = 1. - prior_vec
        loss1 = (ccp * neg_loss_neg_data_vec).sum()
        unmax_loss_vec = pos_loss_unlabel_data_vec - ccp * pos_loss_neg_data_vec
        max_loss_vec = torch.abs(unmax_loss_vec)
        loss2 = max_loss_vec.sum()
        loss = loss1 + loss2
        return loss