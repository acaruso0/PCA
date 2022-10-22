import numpy as np
import scipy.stats as st


def KDE1D(x, entropy=False):
    xmin, xmax = np.min(x), np.max(x)
    xx = np.linspace(xmin, xmax, 100)

    kernel = st.gaussian_kde(x)
    y = kernel(xx)

    if entropy:
        return xx, -np.log2(y)

    return xx, y

def KDE2D(x, y, entropy=False):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    pos = np.vstack([xx.ravel(), yy.ravel()])
    val = np.vstack([x, y])

    kernel = st.gaussian_kde(val)
    z = np.reshape(kernel(pos).T, xx.shape)

    if entropy:
        return xx, yy, -np.log2(z)

    return xx, yy, z

