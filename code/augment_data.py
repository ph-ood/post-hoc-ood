import utils
import shutil
import argparse
import numpy as np
import pandas as pd
from config import *
from tqdm import tqdm
from pathlib import Path
from dataset import ImageDataset
from torchvision import transforms
from augmentations import ShufflePatch
from torch.utils.data import DataLoader

# python3 augment_data.py -d fmnist

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help = "Dataset name", required = True)

if __name__ == "__main__":

    # Set seed, get device
    utils.setSeed(SEED)
    DEVICE = utils.getDevice()

    args = parser.parse_args()
    dname = args.dataset
    patch_size = PATCH_SIZE[dname]
    classes = CLASSES[dname]

    path_data = f"{PATH_DATA}/{dname}"
    path_out = f"{path_data}_patched"
    # Create dirs if they don't exist
    for c in classes:
        for sp in SPLITS:
            Path(f"{path_out}/{sp}/{c}").mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(f"{path_data}/data.csv")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = ImageDataset(
        path_base = path_data, 
        df = df, 
        img_transform = transform, 
        postprocess = False, 
        labelled = False, 
        return_path = True
    )
    dl = DataLoader(ds, batch_size = 128, shuffle = False)

    asp = ShufflePatch(patch_size = patch_size)

    for imgs, paths in tqdm(dl):
        shuf = np.around(255*asp(imgs).permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8) # [b, c, h, w]
        for i in range(shuf.shape[0]):
            shuf_i = shuf[i, ...]
            path = paths[i]
            utils.save(shuf_i, f"{path_out}/{path}")

    # Copy the CSV
    shutil.copy(f"{PATH_DATA}/{dname}/data.csv", f"{path_out}/data.csv")