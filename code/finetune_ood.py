import utils
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from config import *
from tqdm import tqdm
from torch import optim
from copy import deepcopy
from models.vgg16 import VGG16
from dataset import ImageDataset
from losses import DualMarginLoss
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--id", "-i", help = "In-Distribution dataset name", required = True)
parser.add_argument("--ft", "-f", help = "Finetuning dataset name", required = True)
parser.add_argument("--name", "-n", help = "Model name", required = True)
parser.add_argument("--metric", "-m", help = "Model classification metric", required = True)
parser.add_argument("--epoch", "-e", help = "Model epochs trained", required = True)

def train(model, loader, loader_ft, optimizer, criterion):
    
    epoch_loss = 0
    y_true = []
    y_pred = []
    
    model.train()
    # Note: zip() will truncate the longer iterable to the length of the shorter one
    # thus, ensure len(loader_ft) > len(loader). Batch sizes can be kept different for achieving this.
    for (imgs, labels), imgs_ft in tqdm(zip(loader, loader_ft)):

        imgs = imgs.to(DEVICE) # [b, h, w]
        labels = labels.to(DEVICE) # [b,]
        imgs_ft = imgs_ft.to(DEVICE) # [b, h, w]

        optimizer.zero_grad()

        output = model(imgs) # [b, n_classes]
        output_ft = model(imgs_ft) # [b_ft, n_classes]

        preds = output.argmax(dim = -1) # [b,]

        loss = criterion(output, labels, output_ft)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.detach().item()
        y_true += labels.detach().cpu().tolist()
        y_pred += preds.detach().cpu().tolist()

    n_batches = len(loader)
    avg_epoch_loss = epoch_loss / n_batches 

    return avg_epoch_loss, y_true, y_pred

def validate(model, loader, criterion):
    
    epoch_loss = 0
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):

            imgs = imgs.to(DEVICE) # [b, h, w]
            labels = labels.to(DEVICE) # [b,]

            output = model(imgs) # [b, n_classes]
            preds = output.argmax(dim = -1) # [b,]

            loss = criterion(output, labels)
            
            epoch_loss += loss.detach().item()
            y_true += labels.detach().cpu().tolist()
            y_pred += preds.detach().cpu().tolist()

    n_batches = len(loader)
    avg_epoch_loss = epoch_loss / n_batches 

    return avg_epoch_loss, y_true, y_pred

if __name__ == "__main__":  

    # Set seed, get device
    utils.setSeed(SEED)
    DEVICE = utils.getDevice()

    args = parser.parse_args()
    dname = args.id # in-distr. data
    dname_ft = args.ft # ood data
    model_name = args.name
    model_metric = float(args.metric)
    model_epoch = args.epoch

    # Get config for data

    path_model_wt = f"{path_wt}/{model_name}_metric{model_metric:.4f}_epoch{model_epoch}.pt"

    path_data = f"{PATH_DATA}/{dname}"
    path_data_ft = f"{PATH_DATA}/{dname_ft}"
    path_wt = f"{PATH_WT}/{dname}"

    data_mean = DATA_MEAN[dname]
    data_std = DATA_STD[dname]

    data_mean_ft = DATA_MEAN[dname_ft]
    data_std_ft = DATA_STD[dname_ft]

    batch_size = BATCH_SIZE[dname]
    batch_size_ft = BATCH_SIZE[dname_ft]

    lr = LR[dname]
    n_classes = N_CLASSES[dname]
    n_epochs = EPOCHS[dname] 

    # Load csv file
    df = pd.read_csv(f"{path_data}/data.csv")

    # Train and val
    df_train_val = df[df["split"] == "train"]
    df_train, df_val = train_test_split(df_train_val, test_size = 5000, shuffle = True, random_state = SEED)
    df_train.reset_index(inplace = True, drop = True)
    df_val.reset_index(inplace = True, drop = True)

    # Test
    df_test = df[df["split"] == "test"]
    df_test.reset_index(inplace = True, drop = True)

    # Finetuning dataset
    df_ft = pd.read_csv(f"{path_data_ft}/data.csv")

    # Define transforms
    interpolation = transforms.functional.InterpolationMode.BILINEAR
    train_transform = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])
    ft_transform = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
        transforms.ToTensor(),
        transforms.Normalize(data_mean_ft, data_std_ft)
    ])

    # Create dataset objects
    ds_train = ImageDataset(path_base = path_data, df = df_train, img_transform = train_transform)
    ds_val   = ImageDataset(path_base = path_data, df = df_val, img_transform = val_transform)
    ds_test  = ImageDataset(path_base = path_data, df = df_test, img_transform = val_transform)
    ds_ft    = ImageDataset(path_base = path_data_ft, df = df_ft, img_transform = ft_transform, labelled = False)

    # Create dataloaders
    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)
    dl_val   = DataLoader(ds_val, batch_size = batch_size, shuffle = False)
    dl_test   = DataLoader(ds_test, batch_size = batch_size, shuffle = False)
    dl_ft   = DataLoader(ds_ft, batch_size = batch_size_ft, shuffle = True)

    # Define model
    model = VGG16(n_classes = n_classes, fc_dropout = 0.25)
    model.load_state_dict(torch.load(path_model_wt))
    model = model.to(DEVICE)

    # Losses
    criterion = nn.CrossEntropyLoss() # assumes n_classes > 2 for all data used
    criterion = criterion.to(DEVICE)
    criterion_ft = DualMarginLoss(t = 1, m_i = 1, m_o = 1, alpha = 0.1)
    criterion_ft = criterion_ft.to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr = lr
    )

    # Train and validate
    best_epoch = 0
    best_metric = 0
    best_model_state = None
    for epoch in range(n_epochs):
       
        print(f"Epoch: {epoch+1:02}/{n_epochs:02}")

        train_loss, train_true, train_preds = train(model, dl_train, dl_ft, optimizer, criterion_ft)
        val_loss, val_true, val_preds = validate(model, dl_val, criterion)
        
        train_metrics = utils.computeMetrics(train_true, train_preds)
        val_metrics = utils.computeMetrics(val_true, val_preds)

        if val_metrics["f1"] > best_metric:
            best_epoch = epoch + 1
            best_metric = val_metrics["f1"]
            best_model_state = deepcopy(model.state_dict())

        print(f"Train loss: {train_loss:.4f} | Val. loss: {val_loss:.4f}")
        utils.printMetrics(train_metrics)
        utils.printMetrics(val_metrics)
        print()

    print(f"Best epoch: {best_epoch} | Best F1: {best_metric:.4f}")
    print()

    # Test
    print("Testing:")
    model.load_state_dict(best_model_state)
    test_loss, test_true, test_preds = validate(model, dl_test, criterion)
    test_metrics = utils.computeMetrics(test_true, test_preds)
    print(f"Test loss: {test_loss:.4f}")
    utils.printMetrics(test_metrics)

    # Save test predictions to original CSV data file
    df_test = df_test.assign(ft_pred = test_preds)
    df_test.to_csv(f"{path_data}/test/data.csv")

    # Save best model weights
    torch.save(best_model_state, f"{path_wt}/ft_{model.name}_metric{best_metric:.4f}_epoch{n_epochs}.pt")