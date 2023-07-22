import pandas as pd

from GPyS_preprocessor import Preprocessor
from GPyS_prediction import Prediction as Pred


import numpy as np
import scipy
import scipy.linalg as la
from typing import TypeAlias, Union, Optional


vector: TypeAlias = list[float]
matrix: TypeAlias = list[vector]

class LOOCV():
    """
    =============================================================
    leave-one-out cross validation (LOOCV)
    =============================================================

    Setting the hyperparameters by minimizing the LOOCV predictive error (sum of squared errors)
    as a means to optimize a certain criterion.

    For GPS, the LOOCV error are measured in Riemannian distances.

    The hSSDist() is the core function in the LOOCV package for criterion optimization.

    Other important functions includes:
    --> PrAngles() for computing the principal angles between two subspaces
    --> RiemannianDist() for computing the Riemannian distance between two subspaces: 2-norm of principal angles
    --> LOOPredEvd() for computing the LOO prediction: EVD version
    --> LOODistEVD() for computing the LOO Riemannian distances: EVD version
    --> default_length() for computing the rule of thumb length scale


    """

    def __init__(self, X: matrix, sample: matrix, beta: Union[int, float]) -> None:
        """
        :param X: Concatenated orthonormal bases (matrix of size n by kl)*
        :param sample : vector of scalar parameters, or matrix of vector parameters
        :param beta: length-scale of correlation, isotropic (scaler)
                                        or separable (vector)

        * n: ambient dimension of the Euclidean space; k: subspace dimension; l: sample size
        """
        self.sample = np.array(sample)
        self.beta = beta
        self.X = X
        self.get_Preprocessor()
        self.get_dimensions()
        self.get_ones()
        self.get_K_and_K_inv()

    def get_K_and_K_inv(self, sample: Optional[matrix] = None,
                        length_scale: Optional[Union[int, float]] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes squared-exponential correlation matrix and its corresponding inverse
        :param sample : vector of scalar parameters, or matrix of vector parameters
        :param length_scale: length-scale of correlation, isotropic (scaler)
                                        or separable (vector)
        :returns: a correlation matrix K
        """
        if sample is None:
            sample = self.sample
        if length_scale is None:
            length_scale = self.beta
        self.K = Pred.construct_corr_matrix0(sample, length_scale)
        self.K_inv = la.solve(self.K, np.identity(self.K.shape[0]))
        return (self.K, self.K_inv)

    @staticmethod
    def default_length(d: int, l: int) -> float:
        """
        Rule-of-thumb lengthscale of the SE kernel for GPS
        :param d: parameter dimension
        :param l: sample size
        returns: isotropic lengthscale for the unit box [0, 1]^d.
        """
        return (3 * d**(3/2)) / l

    def get_Preprocessor(self) -> np.ndarray:
        """
        :returns: preprocessing quantities of interest
        """
        Preprocessor.setup(self.X)
        self.XtX = Preprocessor.get_XX_cross_product()
        self.VtX = Preprocessor.get_Vt_X()
        return self.XtX, self.VtX

    def get_dimensions(self) -> int:
        """
        :returns: subspace dimension
        """
        _, K_inv = self.get_K_and_K_inv()
        self.k = self.XtX.shape[-1] // K_inv.shape[-1]
        return self.k

    def get_ones(self) -> np.ndarray:
        """
        :returns: matrix of ones (k * k)
        """
        self.Jk = np.ones((self.k, self.k))
        return self.Jk

    # compute Riemannian distance
    def PrAngles(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
         Principal angles between two subspaces
        :param X: subspace representation
        :param Y: subspace representation
        returns: a vector of principal angles in increasing order, [0, pi/2]
        """
        XtY = X.T @ Y
        svdXtY = la.svd(XtY)
        sigma = svdXtY[1]
        sigma[sigma > 1 ] = 1
        theta = np.arccos(sigma)
        #  Correct small principal angles: Grassmann Log per [@Zimmermann2019]
        #  svdXtY[2] is V hence we take the transpose in comparison to R computation (i.e., svdXtY$v)
        M = (X @ svdXtY[0] - (Y @ (svdXtY[2].T @ np.diag(svdXtY[1])))) @ (svdXtY[2])
        svdM = la.svd(M)
        thetaGL19 = np.flip(np.arcsin(svdM[1]))
        isSmall = theta < (0.1 * np.pi / 2)
        theta[isSmall] = thetaGL19[isSmall]
        return theta

    def RiemannianDist(self, X: np.ndarray, Y: np.ndarray, normalize: bool =True) -> float:
        angles = self.PrAngles(X, Y)
        if normalize:
            angles = angles / (np.pi / 2)
        return np.sqrt(np.sum(angles ** 2))

    def LOOPredEvd(self, i, K_inv: Optional[np.ndarray] = None, VtX: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute LOO prediction: EVD version
        :param i (int): LOO index
        :param K_inv (: inverse of correlation of matrix
        :returns:LOO prediction in the global basis, \equation{\circ{V}_{-i}}.
        """
        if K_inv is None:
            K, K_inv = self.get_K_and_K_inv()
        if VtX is None:
            VtX = self.VtX
        # Row / column indices of the i-th training point
        idxi = (i-1) * self.k + np.arange(1, self.k+1) + (self.k-1)
        kbii = K_inv[i, i]
        kbio = np.delete(K_inv[i], [i])
        kbi = np.delete(np.delete(K_inv, i, 0), i, 1)
        Dkbioi = np.diag(1 / kbio)
        Deltai = Dkbioi @ kbi @ Dkbioi * kbii - 1
        Deltai = np.tril(Deltai) + np.triu(Deltai.T, 1)  # force symmetry of Deltai
        # construct \Pi_{-1}, positive semi-definite
        PIi = (np.delete(np.delete(self.XtX, idxi, 0), idxi, 1)) * np.kron(Deltai, self.Jk)
        # add a nugget for numerical stability
        tr = np.sum(abs(np.diag(PIi)))
        PIi += np.finfo('float').eps * np.eye(self.k * (len(self.sample) - 1)) * tr # possible discrepancies in length
        # Construct P_i = A_{-i} (\Pi_{-i})^{-1} A_{-i}^T
        Ami = np.delete(self.VtX, idxi, 1)
        Sol = la.solve(PIi, Ami.T)
        Pi = Ami @ Sol

        # added for the special case of G_{1,2}
        if Pi.shape[0] == 2:
            values, vectors = la.eigh(Pi)
            return vectors[:, np.arange(0, self.k)]
        else:
            values, vectors = scipy.sparse.linalg.eigsh(Pi, self.k, maxiter=10000, tol=1e-16)
            return vectors


    def LOODistEVD(self, i,  K_inv: Optional[np.ndarray] = None, VtX: Optional[np.ndarray] = None) -> float:
        """
         Compute LOO Riemannian distances: EVD version
        :param i (int): LOO index
        :param K_inv : inverse of correlation of matrix
        :param VtX : preprocessing quantity of interest
        :returns: Riemannian distances (normed angle)
        """
        if K_inv is None:
            _, K_inv = self.get_K_and_K_inv()
        if VtX is None:
            VtX = self.VtX
        Vcirc = self.LOOPredEvd(i, K_inv, VtX)
        # Row/colum indices of the i-th training point
        idxi = (i-1) * self.k + np.arange(1, self.k+1) + (self.k - 1)
        return self.RiemannianDist(self.VtX[:, idxi], Vcirc)

    # Criterion to minimize, the LOOCV prediction error: sum of squared Riemannian distances.
    # @note require (thetaTrain, XtX, VbtX; getKinv)

    def hSSDist(self, length: Union[int, float]) -> float:
        """
        Criterion to minimize, the LOOCV prediction error
        :param length: default lengthscale
        :returns: sum of squared Riemannian distances.
        """
        _, K_inv = self.get_K_and_K_inv(length_scale=[length])
        if len(self.sample.shape) > 1:
            l = self.sample.shape[0]
        else:
            l = len(self.sample)
        self.dist = np.array([self.LOODistEVD(i, K_inv=K_inv) for i in np.arange(0, l)]) # note that i needs to be in the function -- to be updated later
        self.objective = np.sum(self.dist**2)
        print(f"[objective: {self.objective}]; parameter: {length}")
        return self.objective


# Testing the functions:
if __name__ == "__main__":

    sample = [[0.00], [0.17], [0.33], [0.50],
              [0.67], [0.83], [1.00]]
    beta = [0.3809524]
    X = [[0.809017, 0.1045284, -0.6691306, -1.0, -0.6691306, 0.1045284, 0.809017],
                        [0.5877852, 0.9945219, 0.7431448, -0.0, -0.7431448, -0.9945219, -0.5877852]]

    LOO = LOOCV(X, sample=sample, beta=beta)

    #Test get_K_and_K_inv()
    K, K_inv = LOO.get_K_and_K_inv()

    #Test default_length ()
    print(LOO.default_length(1, len(sample)))

    # Test get_Preprocessor()
    LOO.get_Preprocessor()

    # Test get_dimensions()
    LOO.get_dimensions()

    # Test get_ones()
    LOO.get_ones()

    #Test LOOPredEvd
    LOO.LOOPredEvd(0)

    #Test LOODistEVD()
    LOO.LOODistEVD(0)

    # Test hSSDist()
    print("hSSDist", LOO.hSSDist(1.44213061))






















