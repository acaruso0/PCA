import numpy as np
import scipy.stats as st

from typing import Tuple


def KDE2D(x: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray]:
    """
    Estimates probability density function (PDF) of a 2D array via kernel density
    estimation.

    Parameters
    ----------
    x, y : np.ndarray
        Input arrays.

    bins : int
        Number of bins for x and y.

    Returns
    -------
    (xx, yy, z) : Tuple[np.ndarray]
        X grid, Y grid, and density estimation.
    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xx, yy = np.mgrid[xmin:xmax:bins*1j, ymin:ymax:bins*1j]

    pos = np.vstack([xx.ravel(), yy.ravel()])
    val = np.vstack([x, y])

    kernel = st.gaussian_kde(val)
    z = np.reshape(kernel(pos).T, xx.shape)

    return xx, yy, z

