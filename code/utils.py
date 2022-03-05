import cv2
import torch
import random
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_theme()
from sklearn.metrics import accuracy_score, f1_score

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getDevice():

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache() # clear cache if non-empty
    else:
        device = "cpu"
  
    dummy = torch.zeros(1, 1)
    try:
        dummy = dummy.to(device)
        if device == "cuda":
            # See which GPU has been allotted 
            print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        elif device == "cpu":
            print("Using CPU")
        else:
            raise ValueError("Invalid device")
    except RuntimeError as re:
        print("Specified GPU not available, using CPU")
        device = "cpu"
    
    del dummy

    return device

def classes2labels(classes):
    return {classes[i] : i for i in range(len(classes))}

def load(path, gray = False):
    if gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image cannot be read: {path}")
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image cannot be read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def computeMetrics(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average = "macro")
    m = {
        "acc" : acc,
        "f1" : f1
    }
    return m

def printMetrics(m):
    keys = sorted(m.keys())
    mt = " | ".join([f"{k} = {m[k]:.4f}" for k in keys])
    print(mt)

def densityPlot(data, title = None, path_save = None):
    ax = sns.kdeplot(data = data, fill = True, alpha = 0.25, palette = "crest")
    sns.move_legend(ax, ncol = 2, loc = "best")
    if title is not None:
        plt.title(title)
    ymin, ymax = ax.get_ylim()
    ymax = ymax + 0.1*ymax
    plt.ylim([ymin, ymax])
    plt.xlabel("Score")
    if path_save is not None:
        plt.savefig(path_save)
        plt.close()
    else:
        plt.show()
