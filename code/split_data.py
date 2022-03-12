import shutil
import argparse
import pandas as pd
from pathlib import Path

# Splits a dataset into the subsets of classes given

parser = argparse.ArgumentParser()
parser.add_argument("--dname", "-d", help = "Dataset name", required = True)
parser.add_argument("--subset_1", "-s1", help = "First subset, format: \"class1, class2, ...\"", required = True)
parser.add_argument("--subset_2", "-s2", help = "Second subset, format: \"class1, class2, ...\"", required = True)

def sub2ext(sub):
    # first char of each class concatendated
    return "".join([s[0] for s in sub]) 

def copyDf(df, path_base_src, path_base_dst):
    for i in range(len(df)):
        row = df.iloc[i]
        fpath = row["path"]
        path_src = f"{path_base_src}/{fpath}"
        path_dst = f"{path_base_dst}/{fpath}"
        shutil.copyfile(path_src, path_dst)

if __name__ == "__main__":
    
    args = parser.parse_args()
    dname = args.dname
    sub1 = [s.trim() for s in args.subset_1.split(",")]
    sub2 = [s.trim() for s in args.subset_2.split(",")]

    ext1 = sub2ext(sub1)
    ext2 = sub2ext(sub2)
    if ext1 == ext2:
        raise ValueError("Both sets of classes yield the same extension")

    # Get config for data
    path_data = f"{PATH_DATA}/{dname}"

    # Output data paths
    path_data1 = f"{path_data}_{ext1}"
    path_data2 = f"{path_data}_{ext2}"

    # Create them if they don't exist
    Path(path_data1).mkdir(parents = True, exist_ok = True)
    Path(path_data2).mkdir(parents = True, exist_ok = True)

    # Load csv file
    df = pd.read_csv(f"{path_data}/data.csv")

    # Get subsets
    df1 = df[df["class"].isin(sub1)]
    df1.reset_index(inplace = True, drop = True)
    df2 = df[df["class"].isin(sub2)]
    df2.reset_index(inplace = True, drop = True)

    if len(df1) == 0:
        raise ValueError("df1 is empty")
    if len(df2) == 0:
        raise ValueError("df2 is empty")

    # Copy everything
    copyDf(df1, path_data, path_data1)
    copyDf(df2, path_data, path_data2)
    
    # Save CSVs
    df1.to_csv(f"{path_data1}/data.csv", index = False)
    df2.to_csv(f"{path_data2}/data.csv", index = False)