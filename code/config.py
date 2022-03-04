SEED = 0

PATH_DATA = "../data"
PATH_WT = "../weights"
PATH_RES = "../results"

SPLITS = ["train", "test"]

CLASSES = {
    "mnist" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "cifar10" : ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}

N_CLASSES = {
    "mnist" : 10,
    "cifar10" : 10
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
    "mnist" : 10,
    "cifar10" : 10
}
