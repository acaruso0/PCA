import math
import numpy as np


def GetFreedmanDiaconis(v: np.ndarray) -> int:
    """
    Applies the Freedman-Diaconis rule to estimate the optimal number of bins.

    Parameters
    ----------
    v : np.ndarray
        Input array to calculate the number of bins.

    Returns
    -------
    nbins : int
        Optimal number of bins.

    References
    ----------
    .. [1] Freedman, D. and Diaconis, P., 1981.
        On the histogram as a density estimator: L 2 theory.
        Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 57(4), pp.453-476.
    """
    N = len(v)

    IQR = np.quantile(v, 0.75) - np.quantile(v, 0.25)
    nbins = math.ceil((max(v) - min(v))/(2*IQR*N**(-1/3)))
    return nbins

def GetMutualInfo(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the mutual information between v1 and v2 using

    I(X,Y) = H(X) + H(Y) - H(X,Y)

    where H is the Shannon entropy calculated by binning v1 and v2 using
    the Freedman-Diaconis rule.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Input arrays to calculate the mutual information.

    Returns
    -------
    I : float
        Mutual information.

    References
    ----------
    .. [1] Bishop, C.M. and Nasrabadi, N.M., 2006.
        Pattern recognition and machine learning (Vol. 4, No. 4, p. 738).
        New York: springer.
    """

    assert(len(v1) == len(v2))

    Bx = GetFreedmanDiaconis(v1)
    By = GetFreedmanDiaconis(v2)
    Bxy = (Bx, By)

    Px = np.histogram(v1, Bx)[0]
    Px = Px/float(np.sum(Px))
    Px = Px[np.nonzero(Px)]
    Hx = -np.sum(Px*np.log2(Px))

    Py = np.histogram(v2, By)[0]
    Py = Py/float(np.sum(Py))
    Py = Py[np.nonzero(Py)]
    Hy = -np.sum(Py*np.log2(Py))

    Pxy = np.histogram2d(v1, v2, Bxy)[0]
    Pxy = Pxy/float(np.sum(Pxy))
    Pxy = Pxy[np.nonzero(Pxy)]
    Hxy = -np.sum(Pxy*np.log2(Pxy))

    I = Hx + Hy - Hxy

    return I

