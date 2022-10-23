import numpy as np

class PCA():
    def __init__(self, data):
        self._data = data
        self._dim = data.shape[1]
        self._eigval, self._eigvec = self._preprocess()
        self._components = self._components()

    def _preprocess(self):
        # Center data onto COM of dataset.
        self._data = self._data - np.mean(self._data, axis=0)

        # Calculate covariance on transposed data. Variables become: columns -> rows.
        cov = np.cov(self._data.T)
        eigval, eigvec = np.linalg.eig(cov)

        # Descending filter.
        filt = np.argsort(eigval)[::-1]
        eigval_sorted = eigval[filt]
        eigvec_sorted = eigvec.T[filt]

        return eigval_sorted, eigvec_sorted

    def _components(self):
        for n in range(self._dim):
            v = self._eigvec[n]/np.linalg.norm(self._eigvec[n])
            PCA_comp = np.dot(self._data, v)

            yield PCA_comp

    def GetComponent(self):
        return next(self._components)

