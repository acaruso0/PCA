import numpy as np
from kde import KDE1D, KDE2D


def GetMutualInfo(v1, v2):
    xx1, y1 = KDE1D(v1)
    Px = np.repeat(y1, 100).reshape(-1,100).flatten()

    xx2, y2 = KDE1D(v2)
    Py = np.repeat(y2, 100).reshape(-1,100).T.flatten()

    xx, yy, z = KDE2D(v1, v2)
    Pxy = z.flatten()

    """
    bins = 100
    Px = np.histogram(v1, bins, density=True)[0]
    Py = np.histogram(v2, bins, density=True)[0]
    Pxy = np.histogram2d(v1, v2, bins, density=True)[0]

    Px = Px[Px > 0]
    Py = Py[Py > 0]
    Pxy = Pxy[Pxy > 0]

    Hx = -np.sum(Px*np.log2(Px))
    Hy = -np.sum(Py*np.log2(Py))
    Hxy = -np.sum(Pxy*np.log2(Pxy))

    I = Hx + Hy - Hxy
    """

    I = 0
    for n in range(Pxy.shape[0]):
        I += Pxy[n]*np.log2(Pxy[n]/(Px[n]*Py[n]))

    return I

