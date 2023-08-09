from GPyS import GPyS_LOOCV_error

from scipy.optimize import minimize
import numpy as np



def optimal_length_scale(X,
                         sample, length=None, scale=1, d=1,):
    """
    Compute optimal lengthscale by minimizing objective/cost function
    :param X: Concatenated orthonormal bases ([[float]])
    :param sample: vector of scalar parameters, or matrix of vector parameters ([[float]])
    :param scale: scale unit box representation of length scale by actual parameter range (float)
    :param length : default length-scale  - (float)
    :param d: parameter dimension to computing default length scale (optional as the user can precompute length)
    :returns: optimal lengthscale
    """
    if length == None:
        length = GPyS_LOOCV_error.LOOCV.default_length(d, l=len(sample)) # unit box representation (to be scaled by parameter range)
    length = length * scale
    print("scaled_lengthscale: ", length)
    LOO = GPyS_LOOCV_error.LOOCV(X, sample=sample, beta=[length])
    # compute bounds for objective function optimization (discretion of user)
    lenUpper = length * 2
    lenLower = length * 0.5
    # perform optimization by minimizing objective function
    ret = minimize(LOO.hSSDist, lenLower, method='Nelder-Mead', bounds=[(lenLower, lenUpper)], tol=1e-6)
    return ret.x


# ____________________________________________________________________________
# see test examples below:
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Example G12 case
    # ------------------------------------------------------------------------
    # Preprocessing quantities (i.e., X and sample) have been previously computed
    # You can refer to the example_G12 file for further info on the computations

    X_example_G12 = [[0.809017, 0.1045284, -0.6691306, -1.0, -0.6691306, 0.1045284, 0.809017],
                        [0.5877852, 0.9945219, 0.7431448, -0.0, -0.7431448, -0.9945219, -0.5877852]]

    sample_example_G12 = [[0.628319], [1.466077], [2.303835], [3.141593],
              [3.979351], [4.817109], [5.654867]]

    # call optimal_length_scale function
    print("example_G12_optimal_length_scale: ", optimal_length_scale(X_example_G12, sample_example_G12,
                                                                     length=0.42857142857142855, scale= 2 * np.pi))

