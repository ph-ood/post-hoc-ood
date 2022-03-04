import sys
import utils
import torch
import numpy as np
import pandas as pd
from torch import nn
from config import *
from tqdm import tqdm
from torch import optim
from copy import deepcopy
from models.vgg16 import VGG16
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def train(model, loader, optimizer, criterion):
    
    epoch_loss = 0
    
    model.train()
    for imgs, labels in tqdm(loader):

        imgs = imgs.to(DEVICE) # [b, h, w]
        labels = labels.to(DEVICE) # [b,]

        optimizer.zero_grad()

        output = model(imgs) # [b, n_classes]

        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.detach().item()

    n_batches = len(loader)
    avg_epoch_loss = epoch_loss / n_batches 

    return avg_epoch_loss

def validate(model, loader, criterion):
    
    epoch_loss = 0
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):

            imgs = imgs.to(DEVICE) # [b, h, w]
            labels = labels.to(DEVICE) # [b,]

            optimizer.zero_grad()

            output = model(imgs) # [b, n_classes]

            loss = criterion(output, labels)
            
            epoch_loss += loss.detach().item()

    n_batches = len(loader)
    avg_epoch_loss = epoch_loss / n_batches 

    return avg_epoch_loss

if __name__ == "__main__":  

    # Set seed, get device
    utils.setSeed(SEED)
    DEVICE = utils.getDevice()

    dname = sys.argv[1]

    # Get config for data
    path_data = f"{PATH_DATA}/{dname}"
    path_wt = f"{PATH_WT}/{dname}"
    batch_size = BATCH_SIZE[dname]
    lr = LR[dname]
    n_classes = N_CLASSES[dname]
    n_epochs = EPOCHS[dname] 

    # Load csv file, get splits

    df = pd.read_csv(f"{path_data}/data.csv")

    df_train = df[df["split"] == "train"]
    df_train.reset_index(inplace = True, drop = True)

    df_val = df[df["split"] == "test"]
    df_val.reset_index(inplace = True, drop = True)

    # Define transforms
    interpolation = transforms.functional.InterpolationMode.BILINEAR
    train_transform = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
        transforms.ToTensor(),
    ])

    # Create dataset objects
    ds_train = ImageDataset(path_base = path_data, df = df_train, img_transform = train_transform)
    ds_val   = ImageDataset(path_base = path_data, df = df_val, img_transform = val_transform)

    # Create dataloaders
    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)
    dl_val   = DataLoader(ds_val, batch_size = batch_size, shuffle = False)

    # Define model
    model = VGG16(n_classes = n_classes, fc_dropout = 0.25)
    model = model.to(DEVICE)

    # Loss
    criterion = nn.CrossEntropyLoss() # assumes n_classes > 2 for all data used
    criterion = criterion.to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr = lr
    )

    for epoch in range(n_epochs):
       
        print(f"Epoch: {epoch+1:02}/{n_epochs:02}")

        train_loss = train(model, dl_train, optimizer, criterion)
        val_loss = validate(model, dl_val, criterion)
        
        print(f"Train loss: {train_loss:.4f} | Val. loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # torch.save(model.state_dict(), f"{path_wt}/{model.name}_epoch{n_epochs}.pt")