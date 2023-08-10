# What is GPyS?
This is a prototypical implementation of Gaussian Process Subspace (GPS) Prediction in the Python programming language. 
For the original research article documenting the method, see the Citation section.

## Table of Contents
* [Citation](#citation)
* [Installation](#installation)
* [Example Use](#example-use)
  * [For GPS Preprocessor](#for-gps-preprocessor)
  * [For GPS Hyperparameter Training](#for-gps-hyperparameter-training)
  * [For GPS Prediction](#for-gps-prediction)

## Citation

- Ruda Zhang, Simon Mak, and David Dunson. Gaussian Process Subspace Prediction for Model Reduction. SIAM Journal on Scientific Computing, 2022. https://epubs.siam.org/doi/10.1137/21M1432739

## Installation

Install the package[^1] via pip using the following command:

- ```pip install GPyS==0.1.2```

## Example Use 

#### After installing the package you can load all modules as shown below:
```
from GPyS import GPyS_preprocessor, GPyS_prediction, GPyS_LOOCV_error
```
#### For GPS Preprocessor:
  - Note that only ```GPyS_preprocessor.Preprocessor.setup(X)``` takes in argument X and this must be called first before any other functions
  - The remaining functions merely return preprocessing quantities of interests

#### For GPS Hyperparameter Training:
  - Utilize ```GPyS_LOOCV_error.LOOCV.hSSDist(length)``` method for the objective function computation at a given (default) length scale
  - Please take a look at the LOOCV_script.py to see an example computation of optimal lengthscale for GPS. 
  - Also, all the functions can be independently called here. 

#### For GPS Prediction:
  - Call ```GPyS_prediction.Prediction.GPS_Prediction()``` to immediately obtain prediction results
  - Also, all the functions can be independently called here. 

[^1]: this package is created and maintained by Ruda Zhang and Taiwo Adebiyi of the UQ-UH Lab.
