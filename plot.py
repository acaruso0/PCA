import numpy as np
from kde import KDE1D, KDE2D


def PlotPCA1D(x, ax, color, label, bins=100, entropy=False):
    lw = 2
    c = color

    hist, bin_edges = np.histogram(x, bins=bins, density=True)
    deltaX = (bin_edges[1] - bin_edges[0])/2
    X_bins = bin_edges[:-1] + deltaX

    if entropy:
        hist = -np.log2(hist)

    ax.bar(X_bins, hist, width=2*deltaX, edgecolor=c, color='None')

    xlim = max(max(x), abs(min(x)))
    xlim += 0.1*xlim
    ax.set_xlim(-xlim, xlim)

    ax.set_xlabel(F"{label}", fontsize=13)
    ax.set_ylabel(r"$-\mathrm{log}_2$" + F"[p({label})]", fontsize=13)

def PlotPCA2D(x, y, ax, bins, entropy=False):
    xx, yy, z = KDE2D(x, y, bins, entropy)

    # Contourf plot
    cfset = ax.contourf(xx, yy, z, cmap='Blues')
    # Contour plot
    cset = ax.contour(xx, yy, z, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(r"$-\mathrm{log}_2$[p(PCA1)]", fontsize=13)
    ax.set_ylabel(r"$-\mathrm{log}_2$[p(PCA2)]", fontsize=13)

    #ax.scatter(x, y, c="k", alpha=0.3)

