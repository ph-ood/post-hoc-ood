import utils
import argparse
import numpy as np
import pandas as pd
from config import *

# Usage: python3 analyze_scores.py -i mnist -o cifar10 -s energy

parser = argparse.ArgumentParser()
parser.add_argument("--id", "-i", help = "In-Distribution dataset name", required = True)
parser.add_argument("--ood", "-o", help = "Out-Of-Distribution dataset name", required = True)
parser.add_argument("--score", "-s", help = "softmax/energy", required = True)
parser.add_argument("--loss", "-l", help="Loss function used", required=True, choices=["DML", "HEL", "MCL", "LOL"])

if __name__ == "__main__":

    utils.setSeed(SEED)

    args = parser.parse_args()
    dname_i = args.id # in-distr. data
    dname_o = args.ood # ood data
    sname = args.score # score name
    loss_name = LOSS[dname_i]

    str_bn = "bn" if USE_BN else "no_bn"
    str_std = "std" if USE_STD else "no_std"
    loss_prefix = "" if loss_name == "ce" else f"{loss_name}_"

    scores_i = np.load(f"{PATH_RES}/raw/ft_{str_bn}_{str_std}_{loss_prefix}{dname_i}_{dname_o}_{sname}_{args.loss}_id.npy")
    scores_o = np.load(f"{PATH_RES}/raw/ft_{str_bn}_{str_std}_{loss_prefix}{dname_i}_{dname_o}_{sname}_{args.loss}_ood.npy")

    # Denisty plots for scores
    data = {f"ID ({dname_i})" : scores_i, f"OOD ({dname_o})" : scores_o}
    utils.densityPlot(data, title = f"{sname.capitalize()} Score", path_save = f"{PATH_PLT}/ft_{str_bn}_{str_std}_{loss_prefix}{dname_i}_{dname_o}_{sname}_{args.loss}_density.png")

    # Metrics for scores (FPR95, AUROC, AUPR)
    metrics = utils.computeOODMetrics(scores_i, scores_o)
    utils.printMetrics(metrics)
