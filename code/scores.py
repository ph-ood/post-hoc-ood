import torch
from torch.nn import functional as F

def softmaxScore(logits):
    # logits: [b, n_classes]
    s, _ = (F.softmax(logits, dim = -1)).max(dim = -1) # [b,]
    return s

def maxLogitScore(logits):
    # logits: [b, n_classes]
    mx, _ = logits.max(dim = -1) # [b,]
    s = mx #torch.exp(mx)
    return s

def energyScore(logits, T = 1):
    # logits: [b, n_classes]
    s = T*torch.logsumexp(logits / T, dim = -1)
    return s

def dirichletScore(logits, alpha):
    smax = F.softmax(logits, dim = -1) # [b, n_classes]
    term1 = torch.lgamma(alpha.sum(dim = -1)) - (torch.lgamma(alpha)).sum(dim = -1)
    term2 = ((alpha - 1)*torch.log(smax)).sum(dim = -1) # [b,]
    logp = term1 + term2
    p = torch.exp(logp)
    return p

if __name__ == "__main__":

    K = 5
    B = 2

    torch.manual_seed(0)
    alphas = torch.ones(K)*0.5
    logits = torch.rand((B, K))

    s = dirichletScore(logits, alphas)
    print(s)