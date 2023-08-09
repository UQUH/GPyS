# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:13:59 2023

@author: taadebi2
"""

from json import dumps
from GPyS import GPyS_prediction



def principal(X, theta, thetanew, lengthscale, thetanew_index=0, t=None):
	# ------------------------------------------------------------------------------------------------
	# please do not print for large matrices (i.e., comment line 18 to 31 of GPyS_prediction_script.py)
	# -------------------------------------------------------------------------------------------------
	print(
		dumps(
			{
				"X": X,
				"i": thetanew_index,
				"thetanew": thetanew,
				"thetanew[i]": thetanew[thetanew_index],
				"theta": theta,
				"lengthscale": lengthscale,
			}, indent=4, ensure_ascii=False, default=str),
		end="\n\n"
	)

	print("target_point:", thetanew[thetanew_index], end="\n\n")

	ret = GPyS_prediction.Prediction.GPS_Prediction(
		X=X,
		sample=theta,
		target=[thetanew[thetanew_index]],
		length_scale=lengthscale,
		t=t
	)
		
	return ret

# see sample input below:(also refer to example_G12 for an iterative 
#                           use of the principal function) 

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
    
    V_circ, sigma2, eps2  = principal(
                                        X, 
                                        sample, 
                                        target, 
                                        length_scale, 
                                        thetanew_index=0)
    print("Vcirc:\n", V_circ, "sigma2:\n,", sigma2, "eps2:\n", eps2)
