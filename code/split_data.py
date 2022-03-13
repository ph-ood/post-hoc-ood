import utils
import shutil
import argparse
import pandas as pd
from config import *
from tqdm import tqdm
from pathlib import Path

# Splits a dataset into the subsets of classes given

parser = argparse.ArgumentParser()
parser.add_argument("--dname", "-d", help = "Dataset name", required = True)
parser.add_argument("--subset", "-s", help = "The subset, format: \"class1, class2, ...\"", required = True)

def sub2ext(sub):
    # first char of each class concatendated
    return "".join([s[0] for s in sub]) 

def copyDf(df, path_base_src, path_base_dst):
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        fpath = row["path"]
        path_src = f"{path_base_src}/{fpath}"
        path_dst = f"{path_base_dst}/{fpath}"
        shutil.copyfile(path_src, path_dst)

if __name__ == "__main__":
    
    args = parser.parse_args()
    dname = args.dname
    sub = sorted([s.strip() for s in args.subset.split(",")])

    # Get config for data
    path_data = f"{PATH_DATA}/{dname}"

    # Output data path
    ext = sub2ext(sub)
    path_sub = f"{path_data}_{ext}"

    # Load csv file
    df = pd.read_csv(f"{path_data}/data.csv")

    # Create dirs if they don't exist
    for s in sub:
        for sp in SPLITS:
            Path(f"{path_sub}/{sp}/{s}").mkdir(parents = True, exist_ok = True)

    # Get subset
    dfs = df[df["class"].astype(str).isin(sub)]
    dfs.reset_index(inplace = True, drop = True)

    class2label = utils.classes2labels(sub)
    dfs["label"] = dfs["class"].apply(lambda x: class2label[str(x)])
    if len(dfs) == 0:
        raise ValueError("Given subset has no samples")

    # Copy everything
    copyDf(dfs, path_data, path_sub)

    # Save CSVs
    dfs.to_csv(f"{path_sub}/data.csv", index = False)