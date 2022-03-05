import utils
import argparse
import numpy as np
import pandas as pd
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--id", "-i", help = "In-Distribution dataset name", required = True)
parser.add_argument("--ood", "-o", help = "Out-Of-Distribution dataset name", required = True)
parser.add_argument("--score", "-s", help = "softmax/energy", required = True)

if __name__ == "__main__":

    utils.setSeed(SEED)

    args = parser.parse_args()
    dname_i = args.id # in-distr. data
    dname_o = args.ood # ood data
    sname = args.score # score name

    scores_i = np.load(f"{PATH_RES}/raw/{dname_i}_{dname_o}_{sname}_id.npy")
    scores_o = np.load(f"{PATH_RES}/raw/{dname_i}_{dname_o}_{sname}_ood.npy")

    # Denisty plots for scores
    utils.densityPlot(scores_i, scores_o, f"{PATH_PLT}/{dname_i}_{dname_o}_{sname}_density.png")

    # TBD: Metrics for scores (FPR95, AUROC, AUPR)
