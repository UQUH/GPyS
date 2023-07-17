Data set description:
Each files is a text file in CSV format, storing an 29008-by-20 matrix with orthonormal columns.
Files names indicate whether they are training data or GPS prediction results, and the associated parameter.
GPS predictions are generated at (SE kenerl) lengthscale 0.36.
If an implementation of the GPS model is correct, the predictions should match in terms of subspace distance: all subspace angles should be close to zero (< 1e-6).
