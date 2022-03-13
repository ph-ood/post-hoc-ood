import sys
import torch
import pandas as pd
from config import *
from tqdm import tqdm
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Computes channel-wise mean and std for a given dataset
# Usage: python3 mean_and_std.py mnist

def meanAndStd(loader):
    n, s, s2 = 0, 0, 0
    for imgs, _ in tqdm(loader):
        b, c, h, w = imgs.shape
        s += imgs.sum((0, 2, 3))
        s2 += (imgs**2).sum((0, 2, 3))
        n += (b*h*w)
    mu = s / n 
    std = torch.sqrt((s2/n) - (mu**2))
    return mu, std

if __name__ == "__main__":

    dname = sys.argv[1]

    path_data = f"{PATH_DATA}/{dname}"

    df = pd.read_csv(f"{path_data}/data.csv")
    df_train = df[df["split"] == "train"]
    df_train.reset_index(inplace = True, drop = True)

    interpolation = transforms.functional.InterpolationMode.BILINEAR
    train_transform = transforms.Compose([
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), interpolation = interpolation),
        transforms.ToTensor(),
    ])

    ds_train = ImageDataset(path_base = path_data, df = df_train, img_transform = train_transform)
    dl_train = DataLoader(ds_train, batch_size = 32, shuffle = False)

    mean, std = meanAndStd(dl_train)
    print(f"Dataset: {dname}")
    print(f"Mean: {mean}")
    print(f"Std.: {std}")



