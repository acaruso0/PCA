import pandas as pd
import matplotlib.pyplot as plt

from settings import Settings
from pca import PCA
from plot import PlotLogPCA1D, PlotLogPCA2D
from information import GetMutualInfo


if __name__ == "__main__":
    settings = Settings()
    data = pd.read_csv(settings.filename, sep="\s+", header=None).to_numpy()

    pca = PCA(data)
    pca1 = pca.GetComponent()
    pca2 = pca.GetComponent()

    fig, ax = plt.subplots(3, 1, figsize=(5, 10),
                           gridspec_kw={'width_ratios': [1],
                                        'height_ratios': [2, 2, 5]},
                           constrained_layout=True)

    PlotLogPCA1D(pca1, ax[0], "dodgerblue", "PCA1", bins=50)
    PlotLogPCA1D(pca2, ax[1], "crimson", "PCA2", bins=50)
    PlotLogPCA2D(pca1, pca2, ax[2], bins=100)

    MI12 = GetMutualInfo(pca1, pca2)
    MI11sq = GetMutualInfo(pca1, pca1*pca1)
    print(F"Mutual information between PCA1 and PCA2: {MI12:.3f} Sh")
    print(F"Mutual information between PCA1 and PCA1^2: {MI11sq:.3f} Sh")

    with open("output.dat", 'w') as out:
        out.write(F"Mutual information between PCA1 and PCA2: {MI12:.3f} Sh\n")
        out.write(F"Mutual information between PCA1 and PCA1^2: {MI11sq:.3f} Sh\n")

    plt.savefig("output.pdf", format="pdf", dpi=300)
    plt.show()

