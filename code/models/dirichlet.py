import torch
from torch import nn

class Dirichlet(nn.Module):

    def __init__(self, n_classes, return_probs = False, alphas = None):
        super(Dirichlet, self).__init__()
        self.return_probs = return_probs
        self.lsoftmax = nn.LogSoftmax(dim = -1)
        if alphas is not None:
            self.alpha = alphas
        else:
            self.alpha = torch.nn.Parameter(torch.ones(n_classes,)) # uniform over the simplex

    def score(self, logits):
        lsmax = self.lsoftmax(logits) # [b, n_classes]
        s = ((self.alpha - 1)*lsmax).sum(dim = -1) # [b,]
        return s

    def forward(self, logits):
        # logits: [b, n_classes]

        lsmax = self.lsoftmax(logits) # [b, n_classes]
        term1 = torch.lgamma(self.alpha.sum(dim = -1)) - (torch.lgamma(self.alpha)).sum(dim = -1)
        term2 = ((self.alpha - 1)*lsmax).sum(dim = -1) # [b,]
        logp = term1 + term2 # [b,]

        if self.return_probs:
            p = torch.exp(logp)
            return p
        else:
            return logp