import math
import numpy as np
from kde import KDE1D, KDE2D


def GetFreedmanDiaconis(v):
    """
    Freedman, D., Diaconis, P.
    On the histogram as a density estimator:L 2 theory.
    Z. Wahrscheinlichkeitstheorie verw Gebiete 57, 453–476 (1981).
    https://doi.org/10.1007/BF01025868
    """
    N = len(v)

    IQR = np.quantile(v, 0.75) - np.quantile(v, 0.25)
    B = (max(v) - min(v))/(2*IQR*N**(-1/3))
    return math.ceil(B)

def GetMutualInfo(v1, v2):
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

    from sklearn.metrics import mutual_info_score

    Pxy = np.histogram2d(v1, v2, Bxy)[0]
    mi = mutual_info_score(None, None, contingency=Pxy)

    print(F"Message: {mi}")

    return I

