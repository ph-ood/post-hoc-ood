import torch
from torch.nn import functional as F

def softmaxScore(logits):
    # logits: [b, n_classes]
    s, _ = (F.softmax(logits, dim = -1)).max(dim = -1) # [b,]
    return s

def energyScore(logits, T = 1):
    # logits: [b, n_classes]
    s = T*torch.logsumexp(logits / T, dim = -1)
    return s