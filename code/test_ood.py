import utils
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from config import *
from tqdm import tqdm
from models.vgg16 import VGG16
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--id", "-i", help = "In-Distribution dataset name")
parser.add_argument("--ood", "-o", help = "Out-Of-Distribution dataset name")
parser.add_argument("--score", "-s", help = "softmax/energy")

def score(output, sname):
    if sname == "softmax":
        s = 0
    elif sname == "energy":
        s = 0
    else:
        raise ValueError("Incorret score name")
    return s

def computeScores(model, loader, sname):
    
    scores = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):

            imgs = imgs.to(DEVICE) # [b, h, w]
            labels = labels.to(DEVICE) # [b,]

            output = model(imgs) # [b, n_classes]
            preds = output.argmax(dim = -1) # [b,]

            # TBD: compute score
            s = score(output, sname)
            scores.append(s)

    return scores

if __name__ == "__main__":  

    # Set seed, get device
    utils.setSeed(SEED)
    DEVICE = utils.getDevice()

    args = parser.parse_args()
    dname_i = args.id # in-distr. data
    dname_o = args.ood # ood data
    sname = args.score # score name

    # Get config for datas
    path_data_i = f"{PATH_DATA}/{dname_i}"
    path_data_o = f"{PATH_DATA}/{dname_i}"
    
    data_mean_i = DATA_MEAN[dname_i]
    data_std_i = DATA_STD[dname_i]

    data_mean_o = DATA_MEAN[dname_o]
    data_std_o = DATA_STD[dname_o]
    
    n_classes = N_CLASSES[dname_i]
    batch_size = BATCH_SIZE[dname_i]
    path_wt = f"{PATH_WT}/{dname_i}"

    # Load csv file
    df_i = pd.read_csv(f"{path_data_i}/data.csv")
    df_o = pd.read_csv(f"{path_data_o}/data.csv")

    # Test
    df_test_i = df_i[df_i["split"] == "test"]
    df_test_i.reset_index(inplace = True, drop = True)

    df_test_o = df_o[df_o["split"] == "test"]
    df_test_o.reset_index(inplace = True, drop = True)

    # Define transforms
    interpolation = transforms.functional.InterpolationMode.BILINEAR
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

    # Create dataset objects
    ds_i  = ImageDataset(path_base = path_data_i, df = df_test_i, img_transform = transform_i)
    ds_o  = ImageDataset(path_base = path_data_o, df = df_test_o, img_transform = transform_o)

    # Create dataloaders
    dl_i = DataLoader(ds_i, batch_size = batch_size, shuffle = False)
    dl_o = DataLoader(ds_o, batch_size = batch_size, shuffle = False)

    # Load model
    model = VGG16(n_classes = n_classes, fc_dropout = 0.25)
    model.load_state_dict(torch.load(path_model_wt))
    model = model.to(DEVICE)
    model.eval()

    # Test
    print("Scoring on In-Distribution:")
    scores_i = computeScores(model, dl_i)
    print()

    print("Scoring on Out-of-Distribution:")
    scores_o = computeScores(model, dl_o)

    # TBD: plot plots and metrics for scores