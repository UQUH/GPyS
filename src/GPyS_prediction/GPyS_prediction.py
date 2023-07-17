import numpy as np
import scipy
import scipy.linalg as la

from src.GPyS_preprocessor import Preprocessor


class Prediction(object):
    """"
  GP Subspace Regression Prediction, Eigen-Decomposition Version
    """

    @staticmethod
    def predict(X, sample, target, length_scale, t=None):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :param t (int): truncation size (optional)
        :returns: GPS Prediction
        """
        Preprocessor.setup(X)
        X = np.array(X)
        sample = np.array(sample)
        target = np.array(target)
        length_scale = np.array(length_scale)  

        XtX = Preprocessor.get_XX_cross_product()
        K = Prediction.construct_corr_matrix0(sample, length_scale)
        cv = Prediction.construct_corr_vector0(sample, target, length_scale)
        l, k = Prediction.__get_dimensions(K, XtX)
        t = t  
        v, diag_v = Prediction.__compute_v_and_diag_vinv(K, cv)
        khat, k_tilda = Prediction.__compute_khat_and_k_tilda(K, diag_v)
        XPX = Prediction.__compute_XPX(l, k, k_tilda, XtX)
        eps2 = Prediction.__compute_noise_variance(v, cv)
        CPC = Prediction.__compute_CPC(X, XPX)
        d1_squared, v_circ = Prediction.__compute_EVD(t, CPC)
        return Prediction.__GPS_Prediction(X, d1_squared, v_circ, eps2)

    @staticmethod
    def construct_corr_matrix0(sample, length_scale):
        """
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: a correlation matrix K
        """
        sample = np.array(sample)
        length_scale = np.array(length_scale)
        X1, X2 = sample, sample
        beta = length_scale

        if len(beta) == 1:
            scale1 = np.identity(X1.shape[0]) * (1 / beta)
            scale2 = np.identity(X2.shape[0]) * (1 / beta)
            theta_scaled = scale1 @ X1
            theta_prime_scaled = scale2 @ X2
        else:
            scale = np.diag(1 / beta)  
            theta_scaled = scale @ X1
            theta_prime_scaled = scale @ X2
        dist = scipy.spatial.distance_matrix(theta_scaled, theta_prime_scaled)
        return np.exp(-(dist ** 2) / 2)

    @staticmethod
    def __get_dimensions(K, XtX):
        """
        :param K: correlation matrix
        :param XtX: X and X-transposed cross product
        :returns: length of sample points (l) and subspace dimension (k)
        """
        l = K.shape[-1]
        k = XtX.shape[-1] // l
        return l, k

    @staticmethod
    def get_dimensions(X, sample, length_scale):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: length of sample points (l) and subspace dimension (k)
        """
        Preprocessor.setup(X)
        XtX = Preprocessor.get_XX_cross_product()
        K = Prediction.construct_corr_matrix0(sample, length_scale)
        l = K.shape[-1]
        k = XtX.shape[-1] // l
        return l, k

    @staticmethod
    def construct_corr_vector0(sample, target, length_scale):
        """
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target: New parameter
        :param length_scale: length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: a covariance vector cv
        """
        X1, X2 = np.array(sample), np.array(target)
        beta = np.array(length_scale)

        if len(beta) == 1:
            scale1 = np.identity(X1.shape[0]) * (1 / beta)
            scale2 = np.identity(X2.shape[0]) * (1 / beta)
            theta_scaled = scale1 @ X1
            theta_prime_scaled = scale2 @ X2
        else:
            scale = np.diag(1 / beta)
            theta_scaled = scale @ X1
            theta_prime_scaled = scale @ X2

        dist = scipy.spatial.distance_matrix(theta_scaled, theta_prime_scaled)
        cv = np.exp(-(dist ** 2) / 2)
        cv = cv.reshape(np.size(cv))
        return cv

    @staticmethod
    def __compute_v_and_diag_vinv(K, cv):
        """
       :param K: correlation matrix
       :param cv: covariance vector
       :returns: v and inverse of diag_v
       """
        v = la.solve(K, cv)
        diag_v = np.diag(1 / v)
        return v, diag_v

    @staticmethod
    def compute_v_and_diag_vinv(sample, target, length_scale):
        """
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): New parameter
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: v and inverse of diag_v
        """
        K = Prediction.construct_corr_matrix0(sample, length_scale)
        cv = Prediction.construct_corr_vector0(sample, target, length_scale)
        v = la.solve(K, cv)
        diag_v = np.diag(1 / v)
        return v, diag_v

    @staticmethod
    def __compute_khat_and_k_tilda(K, diag_v):
        """
        :param K: correlation matrix
        :param diag_v: inverse of diag_v
        :returns: khat and k_tilda
        """
        khat = la.solve(K, diag_v)
        k_tilda = diag_v @ khat
        tr = np.sum(abs(np.diag(k_tilda)))
        k_tilda += np.finfo('float').eps * np.eye(len(K)) * tr  # add nugget for numerical stability
        return khat, k_tilda

    @staticmethod
    def compute_khat_and_k_tilda(sample, target, length_scale):
        """
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): New parameter
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: khat and k_tilda
        """
        K = Prediction.construct_corr_matrix0(sample, length_scale)
        _, diag_v = Prediction.compute_v_and_diag_vinv(sample, target, length_scale)

        khat = la.solve(K, diag_v)
        k_tilda = diag_v @ khat
        tr = np.sum(abs(np.diag(k_tilda)))
        k_tilda += np.finfo('float').eps * np.eye(len(K)) * tr  # add nugget for numerical stability
        return khat, k_tilda

    @staticmethod
    def __compute_XPX(l, k, k_tilda, XtX):
        """
        :param K: correlation matrix
        :param k: subspace dimension
        :param k_tilda: ---------
        :param XtX: X and X-transposed cross product
        :returns: Block matrix structure
        """
        Jk = np.ones((k, k))
        k_Jk = np.kron(k_tilda, Jk)  # kronecker product
        XPX = np.multiply(XtX, k_Jk)  # Hadamard product
        tr = np.sum(abs(np.diag(XPX)))
        XPX += np.finfo('float').eps * np.eye(l * k) * tr
        return XPX

    @staticmethod
    def compute_XPX(X, sample, target, length_scale):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: Block matrix structure
        """
        Preprocessor.setup(X)
        XtX = Preprocessor.get_XX_cross_product()
        l, k = Prediction.get_dimensions(X, sample, length_scale)
        _, k_tilda = Prediction.compute_khat_and_k_tilda(sample, target, length_scale)

        Jk = np.ones((k, k))
        k_Jk = np.kron(k_tilda, Jk)  # kronecker product
        XPX = np.multiply(XtX, k_Jk)  # Hadamard product
        tr = np.sum(abs(np.diag(XPX)))
        XPX += np.finfo('float').eps * np.eye(l * k) * tr
        return XPX

    @staticmethod
    def __XPX_decomposition(XPX):
        """"
        computes cholesky factor
        returns: upper triangular
        """
        U = la.cholesky(XPX)
        # returns upper triangular
        return U

    @staticmethod
    def XPX_decomposition(X, sample, target, length_scale):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        returns: upper triangular Cholesky factor
        """
        XPX = Prediction.compute_XPX(X, sample, target, length_scale)
        U = la.cholesky(XPX)
        # returns upper triangular
        return U

    @staticmethod
    def __compute_noise_variance(v, cv):
        """
        :param v: linear solve of K and diag_v
        :param cv: covariance vector
        :returns: noise variance
        """
        eps2 = 1 - (cv.T @ v)
        return eps2

    @staticmethod
    def compute_noise_variance(sample, target, length_scale):
        """
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: noise variance
        """
        cv = Prediction.construct_corr_vector0(sample, target, length_scale)
        v, _ = Prediction.compute_v_and_diag_vinv(sample, target, length_scale)
        eps2 = 1 - (cv.T @ v)
        return eps2

    @staticmethod
    def __compute_CPC(X, XPX):
        """
        :param X: Concatenated orthonormal bases
        :param XPX: constructed block matrix
        :returns: matrix cross product of tideL
        """
        PC = la.solve(XPX, Preprocessor.get_Xt_V())
        CPC = Preprocessor.get_Vt_X() @ PC
        return CPC

    @staticmethod
    def compute_CPC(X, sample, target, length_scale):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :returns: matrix cross product of tideL
        """
        Preprocessor.setup(X)
        XPX = Prediction.compute_XPX(X, sample, target, length_scale)
        PC = la.solve(XPX, Preprocessor.get_Xt_V())
        CPC = Preprocessor.get_Vt_X() @ PC
        return CPC

    @staticmethod
    def __compute_EVD(t, CPC):
        """
        :param t: truncation size (optional)
        :param CPC: constructed block matrix
        :returns: truncated if t is provided, else returns full EVD
        """
        if t != None:
            d1_squared, v_circ = scipy.sparse.linalg.eigsh(CPC, t)
        else:
            d1_squared, v_circ = scipy.linalg.eigh(CPC)
        return d1_squared, v_circ

    @staticmethod
    def compute_EVD(X, sample, target, length_scale, t=None):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :param t (int): truncation size (optional)
        :returns: truncated if t is provided, else returns full EVD
        """
        CPC = Prediction.compute_CPC(X, sample, target, length_scale)
        if t != None:
            d1_squared, v_circ = scipy.sparse.linalg.eigsh(CPC, t)
        else:
            d1_squared, v_circ = scipy.linalg.eigh(CPC)
        return d1_squared, v_circ

    @staticmethod
    def __GPS_Prediction(X, d1_squared, v_circ, eps2):
        """
        :param X: Concatenated orthonormal bases
        :param d1_squared: EVD values
        :param v_circ: EVD vectors
        :param eps2: noise variance
        :returns: principal directions, principal variances, and noise variance
        """
        v = Preprocessor.get_V_D_Wt()[0]
        GPS_Prediction = [v @ v_circ, d1_squared, eps2]
        return GPS_Prediction

    @staticmethod
    def GPS_Prediction(X, sample, target, length_scale, t=None):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :param sample ([[float]]): vector of scalar parameters, or matrix of vector parameters
        :param target ([[float]]): parameter point for prediction
        :param length_scale ([float]): length-scale of correlation, isotropic (scaler)
                             or separable (vector), defaults to 1
        :param t (int): truncation size (optional)
        :returns: principal directions, principal variances, and noise variance
        """
        return Prediction.predict(
            X=X,
            sample=sample,
            target=target,
            length_scale=length_scale,
            t=t
        )


if __name__ == "__main__":
    X = [[0.809017, 0.1045284, -0.6691306,
          -1.0, -0.6691306, 0.1045284,
          0.809017], [0.5877852, 0.9945219,
                      0.7431448, -0.0, -0.7431448,
                      -0.9945219, -0.5877852]]
    sample = [[0.628319], [1.466077], [2.303835], [3.141593],
              [3.979351], [4.817109], [5.654867]]
    target = [[6.157522]]
    length_scale = [3]

    prediction = Prediction.GPS_Prediction(
        X=X,
        sample=sample,
        target=target,
        length_scale=length_scale,
    )
    # Test GPS_Prediction()
    print("GPS_Prediction: \n", prediction)

    # # Test construct_corr_matrix0()
    # print("construct_corr_matrix_distance*:\n", Prediction.construct_corr_matrix0(sample, length_scale))

    # # Test get_dimensions()
    # l, k = Prediction.get_dimensions(X, sample, length_scale)
    # print("get_dimensions:\n", l, k)

    # # Test construct_corr_vector0()
    # cv = Prediction.construct_corr_vector0(sample, target, length_scale)
    # print("construct_corr_vector_distance*:\n", cv)

    # # Test compute_v_and_diag_vinv()
    # v, diag_v = Prediction.compute_v_and_diag_vinv(sample, target, length_scale)
    # print("compute_v_and_diag_vinv: \n", v, diag_v)

    # # Test compute_khat_and_k_tilda()
    # khat, k_tilda = Prediction.compute_khat_and_k_tilda(sample, target, length_scale)
    # print("compute_khat_and_k_tilda: \n", khat, k_tilda)

    # # Test compute_XPX()
    # XPX = Prediction.compute_XPX(X, sample, target, length_scale)
    # print("compute_XPX: \n", XPX)

    # # Test XPX_decomposition()
    # U = Prediction.XPX_decomposition(X, sample, target, length_scale)
    # print("compute_XPX_decomposition: \n", U)

    # # Test compute_noise_variance()
    # eps2 = Prediction.compute_noise_variance(sample, target, length_scale)
    # print("compute_noise_variance: \n", eps2)

    # # Test compute_CPC()
    # CPC = Prediction.compute_CPC(X, sample, target, length_scale)
    # print("compute_CPC: \n", CPC)

    # # Test compute_EVD()
    d1_squared, v_circ = Prediction.compute_EVD(X, sample, target, length_scale, t=None)
    print("compute_EVD: \n", d1_squared, v_circ)
