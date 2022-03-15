import torch
import scores as sc
from torch import nn
from torch.nn import functional as F

class DualMarginLoss(nn.Module):

    def __init__(self, T, m_i, m_o, alpha):
        super(DualMarginLoss, self).__init__()
        self.T = T
        self.m_i = m_i
        self.m_o = m_o
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, logits_ft):
        
        loss_discr = self.criterion(logits, labels)
        
        e_i = -sc.energyScore(logits, self.T) # [b,]
        e_o = -sc.energyScore(logits_ft, self.T) # [b_ft,]
        loss_energy = (F.relu(e_i - self.m_i)**2).mean() + (F.relu(self.m_o - e_o)**2).mean()
        
        loss = loss_discr + self.alpha*loss_energy

        return loss

class HarmonicEnergyLoss(nn.Module):

    def __init__(self, T, m_i, m_o, alpha):
        super(HarmonicEnergyLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, logits_ft):
        
        loss_discr = self.criterion(logits, labels)
        
        e_i = sc.energyScore(logits, self.T) # [b,]
        e_o = sc.energyScore(logits_ft, self.T) # [b_ft,]
        
        loss_energy = -(2*e_o)/(1 + e_o*e_i)

        loss = loss_discr + self.alpha*loss_energy.mean()

        return loss