import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def ce(sim_matrix):
    logpt = F.log_softmax(sim_matrix, dim=-1)
    logpt = torch.diag(logpt)
    return -logpt

class TotalLoss(nn.Module):
    def __init__(self,):
        super(TotalLoss, self).__init__()
    def forward(self, i2t_sim_matrix, t2i_sim_matrix, subset=None):
        if subset == 'mbank':
            loss = (ce(i2t_sim_matrix) + ce(t2i_sim_matrix)) / 2
            return loss
        elif subset == 'batch':
            i2t_loss = ce(i2t_sim_matrix).mean()
            t2i_loss = ce(t2i_sim_matrix).mean()
            return i2t_loss, t2i_loss







        



