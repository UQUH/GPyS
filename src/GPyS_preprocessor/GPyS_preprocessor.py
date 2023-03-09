import numpy as np
import scipy.linalg as la


class Preprocessor:
    X = None
    XX_cross_product = None
    V, D, Wt = None, None, None
    Vt_X = None
    Xt_V = None

    @classmethod
    def setup(cls, X):
        """
        :param X ([[float]]): Concatenated orthonormal bases
        :returns: X and X-transposed cross product; 
                  SVD of X;
                  Cross product of svdX$D and svdX$Wt, and
                  Transposed Vt_X
        """
        Preprocessor.X = np.array(X)
        Preprocessor.XX_cross_product = Preprocessor.X.T @ Preprocessor.X

        U, s, Vh = la.svd(Preprocessor.X, full_matrices=False)
        V, D, Wt = U, np.diag(s), Vh
        Preprocessor.V = V
        Preprocessor.D = D
        Preprocessor.Wt = Wt

        Vt_X = D @ Wt
        Preprocessor.Vt_X = Vt_X

        Preprocessor.Xt_V = Vt_X.T

    @classmethod
    def get_XX_cross_product(cls):
        return Preprocessor.XX_cross_product

    @classmethod
    def get_V_D_Wt(cls):
        return (Preprocessor.V, Preprocessor.D, Preprocessor.Wt)

    @classmethod
    def get_Vt_X(cls):
        return Preprocessor.Vt_X

    @classmethod
    def get_Xt_V(cls):
        return Preprocessor.Xt_V


if __name__ == "__main__":
    X = [
        [
            8.09016994e-01, 1.04528463e-01, -6.69130606e-01,
            -1.00000000e+00, -6.69130606e-01, 1.04528463e-01,
            8.09016994e-01
        ],
        [
            5.87785252e-01, 9.94521895e-01, 7.43144825e-01,
            1.22464680e-16, -7.43144825e-01, -9.94521895e-01,
            -5.87785252e-01
        ]
    ]

    Preprocessor.setup(X)

    # Test get_XX_cross_product()
    print("get_XX_cross_product():", Preprocessor.get_XX_cross_product(), end="\n\n")

    # Test get_V_D_Wt()
    print("get_V_D_Wt():", Preprocessor.get_V_D_Wt(), end="\n\n")

    # Test get_Vt_X()
    print("get_Vt_X():", Preprocessor.get_Vt_X(), end="\n\n")

    # Test get_Xt_V()
    print("get_Xt_V():", Preprocessor.get_Xt_V(), end="\n\n")