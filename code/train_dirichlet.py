import utils
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from config import *
from tqdm import tqdm
from torch import optim
from models.vgg16 import VGG16
from dataset import ImageDataset
from torchvision import transforms
from models.dirichlet import Dirichlet
from torch.utils.data import DataLoader

# Usage: python3 train_dirichlet.py -n vgg16 -m 0.9911 -e 5 -d mnist

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help = "Dataset name", required = True)
parser.add_argument("--name", "-n", help = "Model name", required = True)
parser.add_argument("--metric", "-m", help = "Model classification metric", required = True)
parser.add_argument("--epoch", "-e", help = "Model epochs trained", required = True)

EPOCHS_DIR = 10
LR_DIR = 1e-3

def train(model, drch, loader, optimizer):

    epoch_loss = 0

    drch.train()
    for imgs, labels in tqdm(loader):

        imgs = imgs.to(DEVICE) # [b, h, w]

        optimizer.zero_grad()

        logits = model(imgs) # [b, n_classes]

        loss_per_sample = drch(logits) # [b,]
        loss = -1*loss_per_sample.mean() # negative log likelhood

        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    n_batches = len(loader)
    avg_epoch_loss = epoch_loss / n_batches 
    return avg_epoch_loss

if __name__ == "__main__":  

    # Set seed, get device
    utils.setSeed(SEED)
    DEVICE = utils.getDevice()

    # Get args
    args = parser.parse_args()
    dname = args.dataset
    model_name = args.name
    model_metric = float(args.metric)
    model_epoch = args.epoch

    # Get config for data
    path_data = f"{PATH_DATA}/{dname}"
    path_wt = f"{PATH_WT}/{dname}"
    data_mean = DATA_MEAN[dname]
    data_std = DATA_STD[dname]
    batch_size = BATCH_SIZE[dname]
    n_classes = N_CLASSES[dname]
    n_epochs = EPOCHS_DIR

    str_bn = "bn" if USE_BN else "no_bn"
    str_std = "std" if USE_STD else "no_std"
    path_model_wt = f"{path_wt}/{model_name}_{str_bn}_{str_std}_metric{model_metric:.4f}_epoch{model_epoch}.pt"

    # Load csv file
    df = pd.read_csv(f"{path_data}/data.csv")

    # Train
    df_train = df[df["split"] == "train"]
    df_train.reset_index(inplace = True, drop = True)

    # Define transforms
    interpolation = transforms.functional.InterpolationMode.BILINEAR
    if USE_STD:
        transform = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor()
        ])

    # Create dataset objects
    ds_train  = ImageDataset(path_base = path_data, df = df_train, img_transform = transform, postprocess = not USE_STD)

    # Create dataloaders
    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)

    # Define model
    model = VGG16(n_classes = n_classes, use_bn = USE_BN, fc_dropout = 0.25)
    model.load_state_dict(torch.load(path_model_wt))
    model = model.to(DEVICE)
    model.eval()
    utils.freeze(model) # Freeze model

    # Dirichlet Model
    drch = Dirichlet(n_classes = n_classes)
    drch = drch.to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(
        drch.parameters(),
        lr = LR_DIR
    )

    # Train
    print("Training Dirichlet")
    
    for epoch in range(n_epochs):

        print(f"Epoch: {epoch+1:02}/{n_epochs:02}")

        train_loss = train(model, drch, dl_train, optimizer)

        print(f"Train loss: {train_loss:.4f}")
        print("Alphas:")
        print(drch.alpha)
        print()