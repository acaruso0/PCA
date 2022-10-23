import numpy as np
from kde import KDE1D, KDE2D


def GetMutualInfo(v1, v2):
    """
    xx1, y1 = KDE1D(v1)
    Px = np.repeat(y1, 50).reshape(-1,50).flatten()

    xx2, y2 = KDE1D(v2)
    Py = np.repeat(y2, 50).reshape(-1,50).T.flatten()

    xx, yy, z = KDE2D(v1, v2)
    Pxy = z.flatten()
    """

    assert(len(v1) == len(v2))
    N = len(v1)

    # Hacine-Gharbi et al. (2012)
    xi = ( 8 + 324*N + 12*( 36*N + 729*N**2 )**(1/2) )**(1/3)
    Bx = round(xi/6 + 2/(3*xi) + 1/3)
    By = Bx
    print(Bx)

    # Hacine-Gharbi and Ravier (2018)
    cov = np.cov(np.stack((v1, v2)))[0, 1]
    rho = cov/(np.std(v1)*np.std(v2))
    Bxy = round((1/2**(1/2))*(1 + (1 + 24*N/(1 - rho**2))**(1/2))**(1/2))
    print(Bxy)

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

    #I = 0
    #for n in range(Pxy.shape[0]):
    #    I += Pxy[n]*np.log2(Pxy[n]/(Px[n]*Py[n]))

    return I

