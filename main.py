import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pca import PCA
from plot import PlotPCA1D, PlotPCA2D


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
    #ax[0].bar(X_bins1, -np.log(hist1), width=2*deltaX1, edgecolor="dodgerblue", color='None')
    #ax[1].bar(X_bins2, -np.log(hist2), width=2*deltaX2, edgecolor="crimson", color='None')

    PlotPCA1D(pca1, ax[0], "dodgerblue", "PCA1", entropy=True)
    PlotPCA1D(pca2, ax[1], "crimson", "PCA2", entropy=True)
    PlotPCA2D(pca1, pca2, ax[2], entropy=True)

    plt.show()

