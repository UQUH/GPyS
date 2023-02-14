This is a prototypical implementation of GPS in the Python programming language. 
For the original research article documenting the method, see the Citation section.

## Citation

- Ruda Zhang, Simon Mak, and David Dunson. Gaussian Process Subspace Prediction for Model Reduction. SIAM Journal on Scientific Computing, 2022. https://epubs.siam.org/doi/10.1137/21M1432739

## Installation

Install the package via pip using the following command:

- pip install GPyS==0.0.2

## Example Use 

After installing the package you can load it via: 

#### For GPS Preprocessor: 
  - from GPyS_preprocessor import Preprocessor
  - Note that only Preprocessor.setup(X) takes in argument X and this must be called first before any other functions
  - The remaining functions merely returns preprocessing quantities of interests

#### For GPS Prediction: 
  - from GPyS_prediction import Prediction
  - All the functions can be independently called here. 
  - Also, user can directly call Prediction.GPS_Prediction() to immediately obtain prediction results
