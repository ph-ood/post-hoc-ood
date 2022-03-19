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
from torch.utils.data import DataLoader

# Usage: python3 test_classifier.py -n vgg16 -m 0.9911 -e 5 -d mnist

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help = "Dataset name", required = True)
parser.add_argument("--name", "-n", help = "Model name", required = True)
parser.add_argument("--metric", "-m", help = "Model classification metric", required = True)
parser.add_argument("--epoch", "-e", help = "Model epochs trained", required = True)

def validate(model, loader):
    
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):

            imgs = imgs.to(DEVICE) # [b, h, w]
            labels = labels.to(DEVICE) # [b,]

            output = model(imgs) # [b, n_classes]
            preds = output.argmax(dim = -1) # [b,]
            
            y_true += labels.detach().cpu().tolist()
            y_pred += preds.detach().cpu().tolist()

    return y_true, y_pred

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

    str_bn = "bn" if USE_BN else "no_bn"
    str_std = "std" if USE_STD else "no_std"
    path_model_wt = f"{path_wt}/{model_name}_{str_bn}_{str_std}_metric{model_metric:.4f}_epoch{model_epoch}.pt"

    # Load csv file
    df = pd.read_csv(f"{path_data}/data.csv")

    # Test
    df_test = df[df["split"] == "test"]
    df_test.reset_index(inplace = True, drop = True)

    # Define transforms
    interpolation = transforms.functional.InterpolationMode.BILINEAR
    if USE_STD:
        val_transform = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
            transforms.ToTensor()
        ])

    # Create dataset objects
    ds_test  = ImageDataset(path_base = path_data, df = df_test, img_transform = val_transform, postprocess = not USE_STD)

    # Create dataloaders
    dl_test   = DataLoader(ds_test, batch_size = batch_size, shuffle = False)

    # Define model
    model = VGG16(n_classes = n_classes, use_bn = USE_BN, fc_dropout = 0.25)
    model.load_state_dict(torch.load(path_model_wt))
    model = model.to(DEVICE)

    # Test
    print("Testing:")
    test_true, test_preds = validate(model, dl_test)
    test_metrics = utils.computeMetrics(test_true, test_preds)
    utils.printMetrics(test_metrics)

    # Save test predictions to original CSV data file
    df_test = df_test.assign(pred = test_preds)
    df_test.to_csv(f"{path_data}/test/data.csv")