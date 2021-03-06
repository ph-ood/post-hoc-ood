SEED = 0

PATH_DATA = "../data"
PATH_WT = "../weights"
PATH_RES = "../results"
PATH_PLT = f"{PATH_RES}/plots"
PATH_EXAMPLES = "../examples"

SPLITS = ["train", "test"]

CLASSES = {
    "mnist" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "fmnist" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "cifar10" : ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "mnist_35689" : ["3", "5", "6", "8", "9"]
}

N_CLASSES = {
    "mnist" : 10,
    "fmnist" : 10,
    "fmnist_patched" : 10,
    "cifar10" : 10,
    "cifar10_patched" : 10,
    "mnist_01247" : 5,
    "mnist_35689" : 5,
    "mnist_35689_patched" : 5,
    "mnist_patched" : 10
}

USE_BN = True
USE_STD = False

DATA_MEAN = {
    "mnist" : [0.1309, 0.1309, 0.1309],
    "fmnist" : [0.2856, 0.2856, 0.2856],
    "fmnist_patched" : [0.2856, 0.2856, 0.2856],
    "cifar10" : [0.4914, 0.4822, 0.4465],
    "cifar10_patched" : [0.4914, 0.4822, 0.4465],
    "mnist_01247" : [0.1256, 0.1256, 0.1256],
    "mnist_35689": [0.1365, 0.1365, 0.1365],
    "mnist_35689_patched" : [0.1365, 0.1365, 0.1365],
    "mnist_patched" : [0.1309, 0.1309, 0.1309]
}

DATA_STD = {
    "mnist" : [0.2893, 0.2893, 0.2893],
    "fmnist" : [0.3385, 0.3385, 0.3385],
    "fmnist_patched" : [0.3385, 0.3385, 0.3385],
    "cifar10" : [0.2470, 0.2435, 0.2616],
    "cifar10_patched" : [0.2470, 0.2435, 0.2616],
    "mnist_01247" : [0.2859, 0.2859, 0.2859],
    "mnist_35689" : [0.2927, 0.2927, 0.2927],
    "mnist_35689_patched" : [0.2927, 0.2927, 0.2927],
    "mnist_patched" : [0.2893, 0.2893, 0.2893]
}

IMG_SIZE = 32

LOSS = {
    "mnist" : "ce",
    "fmnist" : "ce",
    "fmnist_patched" : "ce",
    "cifar10" : "ce",
    "cifar10_patched" : "ce",
    "mnist_01247" : "ce",
    "mnist_35689" : "ce",
    "mnist_35689_patched" : "ce",
    "mnist_patched" : "ce"
}

BATCH_SIZE = {
    "mnist" : 128,
    "fmnist" : 128,
    "fmnist_patched" : 128,
    "cifar10" : 128,
    "cifar10_patched" : 128,
    "mnist_01247" : 128,
    "mnist_35689" : 128,
    "mnist_35689_patched" : 128,
    "mnist_patched" : 128
}

LR = {
    "mnist" : 1e-4,
    "fmnist" : 1e-4,
    "fmnist_patched" : 1e-4,
    "cifar10" : 1e-4,
    "cifar10_patched" : 1e-4,
    "mnist_01247" : 1e-4,
    "mnist_35689" : 1e-4,
    "mnist_35689_patched" : 1e-4,
    "mnist_patched" : 1e-4
}

EPOCHS = {
    "mnist" : 5,
    "fmnist" : 10,
    "fmnist_patched" : 10,
    "cifar10" : 50,
    "cifar10_patched" : 50,
    "mnist_01247" : 5,
    "mnist_35689" : 5,
    "mnist_35689_patched" : 5,
    "mnist_patched" : 5,
}

PATCH_SIZE = {
    "mnist" : 7,
    "fmnist" : 7,
    "cifar10" : 8,
    "mnist_01247" : 7,
    "mnist_35689" : 7
}

EPOCHS_FT = {
    "mnist" : 5,
    "fmnist" : 10,
    "fmnist_patched" : 10,
    "cifar10" : 10,
    "cifar10_patched" : 10,
    "mnist_01247" : 5,
    "mnist_35689" : 5,
    "mnist_35689_patched" : 5,
    "mnist_patched" : 5,
}

# Specifies (m_in, m_out) values
DML_PARAMS = {
    "mnist" : (-11, -3),
    "fmnist" : (-11, -3),
    "cifar10" : (-11, -3),
    "mnist_35689" : (-11, -3)
}