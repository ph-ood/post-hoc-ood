import utils
import torch
import argparse
import numpy as np
import pandas as pd
import scores as sc
from torch import nn
from config import *
from tqdm import tqdm
from models.vgg16 import VGG16
from dataset import ImageDataset
from torchvision import transforms
from torch.nn import functional as F
from models.dirichlet import Dirichlet
from torch.utils.data import DataLoader

# Usage: python3 average_energy.py -n vgg16 -m 0.9911 -e 5 -i mnist -f cifar10
parser = argparse.ArgumentParser()
parser.add_argument("--id", "-i", help = "In-Distribution dataset name", required = True)
parser.add_argument("--ftd", "-f", help = "Fine-tune dataset name", required = True)
parser.add_argument("--name", "-n", help = "Model name", required = True)
parser.add_argument("--metric", "-m", help = "Model classification metric", required = True)
parser.add_argument("--epoch", "-e", help = "Model epochs trained", required = True)

def score(logits, sname):
    # logits: [b, n_classes]
    T = 1
    s = -sc.energyScore(logits, T) # [b,]

    return s

def computeScores(model, loader, sname):
    
    scores = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):

            imgs = imgs.to(DEVICE) # [b, h, w]
            labels = labels.to(DEVICE) # [b,]

            logits = model(imgs) # [b, n_classes]

            sb = score(logits, sname)
            scores += sb.detach().cpu().tolist()

    return scores

if __name__ == "__main__":  

    # Set seed, get device
    utils.setSeed(SEED)
    DEVICE = utils.getDevice()

    args = parser.parse_args()
    dname_i = args.id # in-distr. data
    dname_o = args.ftd # ood data
    sname = "energy" # score name

    model_name = args.name
    model_metric = float(args.metric)
    model_epoch = args.epoch

    # Get config for datas
    path_data_i = f"{PATH_DATA}/{dname_i}"
    path_data_o = f"{PATH_DATA}/{dname_o}"
    
    data_mean_i = DATA_MEAN[dname_i]
    data_std_i = DATA_STD[dname_i]

    data_mean_o = DATA_MEAN[dname_o]
    data_std_o = DATA_STD[dname_o]
    
    n_classes = N_CLASSES[dname_i]
    batch_size = BATCH_SIZE[dname_i]
    path_wt = f"{PATH_WT}/{dname_i}"

    loss_name = LOSS[dname_i]

    str_bn = "bn" if USE_BN else "no_bn"
    str_std = "std" if USE_STD else "no_std"
    loss_prefix = "" if loss_name == "ce" else f"{loss_name}_"
    #path_model_wt = f"{path_wt}/ft_{model_name}_{str_bn}_{str_std}_{loss_prefix}metric{model_metric:.4f}_epoch{model_epoch}_{args.loss}.pt"
    path_model_wt = f"{path_wt}/{model_name}_{str_bn}_{str_std}_{loss_prefix}metric{model_metric:.4}_epoch{model_epoch}.pt"


    # Load csv file
    df_i = pd.read_csv(f"{path_data_i}/data.csv")
    df_o = pd.read_csv(f"{path_data_o}/data.csv")

    # Test
    df_test_i = df_i[df_i["split"] == "train"]
    df_test_i.reset_index(inplace = True, drop = True)

    df_test_o = df_o[df_o["split"] == "train"]
    df_test_o.reset_index(inplace = True, drop = True)

    # Define transforms
    interpolation = transforms.functional.InterpolationMode.BILINEAR
    if USE_STD:
        transform_i = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor(),
            transforms.Normalize(data_mean_i, data_std_i)
        ])
        transform_o = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor(),
            transforms.Normalize(data_mean_o, data_std_o)
        ])
    else:
        transform_i = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor(),
        ])
        transform_o = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor()
        ])

    # Create dataset objects
    ds_i  = ImageDataset(path_base = path_data_i, df = df_test_i, img_transform = transform_i, postprocess = not USE_STD)
    ds_o  = ImageDataset(path_base = path_data_o, df = df_test_o, img_transform = transform_o, postprocess = not USE_STD)

    # Create dataloaders
    dl_i = DataLoader(ds_i, batch_size = batch_size, shuffle = False)
    dl_o = DataLoader(ds_o, batch_size = batch_size, shuffle = False)

    # Load model
    model = VGG16(n_classes = n_classes, use_bn = USE_BN, fc_dropout = 0.25)
    model.load_state_dict(torch.load(path_model_wt))
    model = model.to(DEVICE)
    model.eval()

    # Test
    print("Computing In-Distribution Scores")
    scores_i = computeScores(model, dl_i, sname)
    print("E_in:", np.mean(scores_i))

    print("Computing Out-Of-Distribution Scores")
    scores_o = computeScores(model, dl_o, sname)
    print("E_out:", np.mean(scores_o))


    # Save scores