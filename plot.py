import numpy as np
import matplotlib.pyplot as plt

from kde import KDE2D


def PlotLogPCA1D(x: np.ndarray, ax: plt.Axes, c: str, lbl: str, bins: int) -> None:
    """
    Plots the negative logarithm in base 2 of the 2-dimensional histogram of the data.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    ax : plt.Axes
        Plotting Axis.

    c : str
        Histogram color.

    lbl: str
        Label of the variable.

    bins : int
        Number of bins for x and y.
    """
    lw = 2

    hist, bin_edges = np.histogram(x, bins=bins)
    hist = hist/float(np.sum(hist))
    hist[np.nonzero(hist)] = -np.log2(hist[np.nonzero(hist)])

    deltaX = (bin_edges[1] - bin_edges[0])/2
    X_bins = bin_edges[:-1] + deltaX

    ax.bar(X_bins, hist, width=2*deltaX, edgecolor=c, color='None')

    xlim = max(max(x), abs(min(x)))
    xlim += 0.1*xlim
    ax.set_xlim(-xlim, xlim)

    ax.set_xlabel(F"{lbl}", fontsize=13)
    ax.set_ylabel(r"$-\mathrm{log}_2$" + F"[p({lbl})]", fontsize=13)

def PlotLogPCA2D(x, y, ax, bins):
    """
    Plots the negative logarithm in base 2 of the 2-dimensional histogram of the data.

    Parameters
    ----------
    x, y : np.ndarray
        Input arrays.

    ax : plt.Axes
        Plotting Axis.

    bins : int
        Number of bins for x and y.
    """
    xx, yy, z = KDE2D(x, y, bins)
    logz = -np.log2(z)

    # Contourf plot
    cfset = ax.contourf(xx, yy, logz, cmap='Blues')
    # Contour plot
    cset = ax.contour(xx, yy, logz, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel("PCA1", fontsize=13)
    ax.set_ylabel("PCA2", fontsize=13)

