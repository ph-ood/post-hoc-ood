PATH_DATA = "../data"

SPLITS = ["train", "test"]

CLASSES = {
    "mnist" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "cifar10" : ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
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
