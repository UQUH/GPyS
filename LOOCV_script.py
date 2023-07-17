from GPyS_LOOCV_error import LOOCV

from scipy.optimize import minimize



def optimal_length_scale(X,
                         sample, length=None, d=1):
    """
    Compute optimal lengthscale by minimizing objective/cost function
    :param X: Concatenated orthonormal bases ([[float]])
    :param sample: vector of scalar parameters, or matrix of vector parameters ([[float]])
    :param length : default length-scale  - (int)
    :param d: parameter dimension to computing default length scale (optional as the user can precompute length)
    :returns: optimal lengthscale
    """
    if length == None:
        length = LOOCV.default_length(d, l=len(sample))
    LOO = LOOCV(X, sample=sample, beta=[length])
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
    print("example_G12_optimal_length_scale: ", optimal_length_scale(X_example_G12, sample_example_G12, 10.0530966))

