SEED = 0

PATH_DATA = "../data"
PATH_WT = "../weights"
PATH_RES = "../results"
PATH_PLT = f"{PATH_RES}/plots"

SPLITS = ["train", "test"]

CLASSES = {
    "mnist" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "cifar10" : ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}

N_CLASSES = {
    "mnist" : 10,
    "cifar10" : 10
}

DATA_MEAN = {
    "mnist" : [0.1309, 0.1309, 0.1309],
    "cifar10" : [0.4914, 0.4822, 0.4465],
}

DATA_STD = {
    "mnist" : [0.2893, 0.2893, 0.2893],
    "cifar10" : [0.2470, 0.2435, 0.2616],
}

IMG_SIZE = 32

BATCH_SIZE = {
    "mnist" : 128,
    "cifar10" : 128,
}

LR = {
    "mnist" : 1e-4,
    "cifar10" : 1e-4
}

EPOCHS = {
    "mnist" : 2,
    "cifar10" : 2
}
