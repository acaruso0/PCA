import numpy as np

from typing import Tuple, Iterable

class PCA():
    def __init__(self, data: np.ndarray):
        self._data = data
        self._dim = data.shape[1]
        self._eigval, self._eigvec = self._preprocess()
        self._components = self._components()

    def _preprocess(self) -> Tuple[np.ndarray]:
        """
 +      Obtains sorted arrays with eigenvalues and eigenvectors of the covariance
        matrix of the variables in the dataset.
 +
 +      Returns
 +      -------
 +      (eigval_sorted, eigvec_sorted) : Tuple[np.ndarray]
 +          Sorted eigenvalues and eigenvectors.
        """
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

    def _components(self) -> Iterable[np.ndarray]:
        """
 +      Yields the normalized components of the PCA.
 +
 +      Yields
 +      -------
        PCA_comp : np.ndarray
 +          PCA component.
        """
        for n in range(self._dim):
            v = self._eigvec[n]/np.linalg.norm(self._eigvec[n])
            PCA_comp = np.dot(self._data, v)

            yield PCA_comp

    def GetComponent(self) -> np.ndarray:
        """
 +      Component getter.
 +
 +      Returns
 +      -------
        next(self._components)
 +          Next component of the PCA.
        """
        return next(self._components)

