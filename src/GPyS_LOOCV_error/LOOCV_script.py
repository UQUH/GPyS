from GPyS_LOOCV_error import LOOCV
from GPyS_anemometer import Anemometer

from scipy.optimize import minimize



def optimal_length_scale(X,
                         sample, length=None, d=1):
    """
    Compute optimal lengthscale by minimizing objective / cost function
    :param X: Concatenated orthonormal bases ([[float]])
    :param sample: vector of scalar parameters, or matrix of vector parameters ([[float]])
    :param length : length-scale of correlation, isotropic (scaler)
                                        or separable (vector) - (int)
    :param d: parameter dimension to compute default length scale
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

    # -------------------------------------------------------------------------
    # Anemometer Case
    # ------------------------------------------------------------------------
    # note that the input data have been pre-computed using the underlying physics 
    # In this case, the data are directly generated and enumerated from series of CSV files (lines 57 - 68)
    path_to_files = "./GPyS_Anenometer/"

    X_train, train_theta = Anemometer.get_X_and_theta(
        path_to_files=path_to_files,
        file_filter="train",
        no_of_files_to_work_with=-1,
        filter_file_from_position_start=None,  # None => start from file 0
        filter_file_from_position_end=None,  # None => goes all the way to the end of file
        no_of_rows_in_files_to_work_with=-1,
        merge_all_x=True,
        )
    sample_anemometer = [[0.00], [0.17], [0.33], [0.50],
              [0.67], [0.83], [1.00]]
    X_anemometer = X_train
    # call optimal_length_scale function
    print("anemometer_optimal_length_scale: ", optimal_length_scale(X_anemometer, sample_anemometer))
