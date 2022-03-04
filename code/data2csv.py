import sys
import utils
import pandas as pd
from config import *
from glob import glob

# Creates a CSV file for a given data folder

dname = sys.argv[1]

dct = {
    "path" : [],
    "split" : [],
    "class" : [],
    "label" : [] 
}

path = f"{PATH_DATA}/{dname}"
classes = CLASSES[dname]
class2label = utils.classes2labels(classes)

for s in SPLITS:
    for c in classes:
        l = class2label[c]
        fpaths = sorted(glob(f"{path}/{s}/{c}/*.png"))
        for fpath in fpaths:
            dct["split"].append(s)
            dct["class"].append(c)
            dct["label"].append(l)
            fname = fpath.replace(f"{path}/", "")
            dct["path"].append(fname)

df = pd.DataFrame.from_dict(dct)
size_train = len(df[df.split == "train"])
size_test = len(df[df.split == "test"])
print(f"Sizes: train: {size_train}, test: {size_test}")
df.to_csv(f"{path}/data.csv", index = False)
