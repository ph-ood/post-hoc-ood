import torch
from torch import nn
from torch.nn import functional as F

class DualMarginLoss(nn.Module):

    def __init__(self, t, m_i, m_o, alpha):
        super(DualMarginLoss, self).__init__()
        self.t = t
        self.m_i = m_i
        self.m_o = m_o
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, logits_ft):
        
        loss_discr = self.criterion(logits, labels)
        
        e_i = -self.t * torch.logsumexp(logits / self.t, dim = -1) # [b,]
        e_o = -self.t * torch.logsumexp(logits_ft / self.t, dim = -1) # [b_ft,]
        loss_energy = (F.relu(e_i - self.m_i)**2).mean() + (F.relu(self.m_o - e_o)**2).mean()
        
        loss = loss_discr + self.alpha*loss_energy

        return loss