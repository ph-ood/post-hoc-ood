import torch
from torch import nn

class Dirichlet(nn.Module):

    def __init__(self, n_classes, return_probs = False):
        super(Dirichlet, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(n_classes,)) # uniform over the simplex
        self.return_probs = return_probs
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, logits):
        # logits: [b, n_classes]

        smax = self.softmax(logits) # [b, n_classes]
        term1 = torch.lgamma(self.alpha.sum(dim = -1)) - (torch.lgamma(self.alpha)).sum(dim = -1)
        term2 = ((self.alpha - 1)*torch.log(smax)).sum(dim = -1) # [b,]
        logp = term1 + term2 # [b,]

        if self.return_probs:
            p = torch.exp(logp)
            return p
        else:
            return logp