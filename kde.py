import numpy as np
import scipy.stats as st


def KDE2D(x, y, bins):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xx, yy = np.mgrid[xmin:xmax:bins*1j, ymin:ymax:bins*1j]

    pos = np.vstack([xx.ravel(), yy.ravel()])
    val = np.vstack([x, y])

    kernel = st.gaussian_kde(val)
    z = np.reshape(kernel(pos).T, xx.shape)

    return xx, yy, z

