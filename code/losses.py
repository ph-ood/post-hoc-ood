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

    def forward(self, logits, labels, logits_ft = None):
        
        loss_discr = self.criterion(logits, labels)
        
        if logits_ft is None:
            loss_energy = 0
        else:
            e_i = -sc.energyScore(logits, self.T) # [b,]
            e_o = -sc.energyScore(logits_ft, self.T) # [b_ft,]
            loss_energy = (F.relu(e_i - self.m_i)**2).mean() + (F.relu(self.m_o - e_o)**2).mean()
        
        loss = loss_discr + self.alpha*loss_energy

        return loss

class HarmonicEnergyLoss(nn.Module):

    def __init__(self, T, alpha):
        super(HarmonicEnergyLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, logits_ft):
        
        loss_discr = self.criterion(logits, labels)
        if logits_ft is None:
            loss_energy = 0
        else:
            e_i = -sc.energyScore(logits, self.T) # [b,]
            e_o = -sc.energyScore(logits_ft, self.T) # [b_ft,]
            
            loss_energy = -(2*e_o.mean())/(1 + e_o.mean()*e_i.mean())

        loss = loss_discr + self.alpha*loss_energy

        return loss


class BrierScore(nn.Module):

    def __init__(self, n_classes):
        super(BrierScore, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
        self.n_classes = n_classes

    def forward(self, logits, labels):
        # logits: [b, n_classes], labels: [b,]
        probs = self.softmax(logits)
        ohes = F.one_hot(labels, num_classes = self.n_classes)
        loss = ((probs - ohes)**2).mean()
        return loss


class MCELoss(nn.Module):
    def __init__(self, T, alpha):
        super(MCELoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, logits_ft):
        loss_discr = self.criterion(logits, labels)

        if logits_ft is None:
            loss_energy = 0
        else:
            e_i = -sc.energyScore(logits, self.T) # [b,]
            e_o = -sc.energyScore(logits_ft, self.T) # [b_ft,]
            loss_energy = torch.reciprocal(1 + torch.exp(-e_i.mean()+e_o.mean()))
        loss = loss_discr + self.alpha * loss_energy
        return loss

class LogLoss(nn.Module):
    def __init__(self, T, alpha):
        super(LogLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, logits_ft):
        loss_discr = self.criterion(logits, labels)

        if logits_ft is None:
            loss_energy = 0
        else:
            e_i = -sc.energyScore(logits, self.T) # [b,]
            e_o = -sc.energyScore(logits_ft, self.T) # [b_ft,]
            loss_energy = torch.log(1 + torch.exp(e_i.mean()-e_o.mean()))
        loss = loss_discr + self.alpha * loss_energy
        return loss
