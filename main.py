import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pca import PCA
from plot import PlotPCA1D, PlotPCA2D
from shannon import GetMutualInfo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute first and second PCA components for a given dataset.")
    parser.add_argument("-i", type=str,
                        help="Input file containing the dataset to analyze.")
    args = parser.parse_args()

    filename = args.i
    data = pd.read_csv(filename, sep="\s+", header=None).to_numpy()

    pca = PCA(data)
    pca1 = pca.GetComponent()
    pca2 = pca.GetComponent()

    fig, ax = plt.subplots(3, 1, figsize=(6, 10),
                           gridspec_kw={'width_ratios': [1],
                                        'height_ratios': [2, 2, 5]})

    PlotPCA1D(pca1, ax[0], "dodgerblue", "PCA1", bins=50, entropy=True)
    PlotPCA1D(pca2, ax[1], "crimson", "PCA2", bins=50, entropy=True)
    PlotPCA2D(pca1, pca2, ax[2], bins=50, entropy=True)

    MI12 = GetMutualInfo(pca1, pca2)
    MI11sq = GetMutualInfo(pca1, pca1*pca1)
    print(MI12)
    print(MI11sq)

    plt.show()

