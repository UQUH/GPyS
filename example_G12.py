# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:33:08 2022

@author: taadebi2
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GPyS_preprocessor import Preprocessor
from GPyS_prediction import Prediction 
import matplotlib.pyplot as plt



class G12(object):

    def __init__(self, n, c, n_new):
        self.n = n
        self.c = c
        self.n_new = n_new

    def set_pdf(self):
        self.DTpdf = pd.DataFrame(np.linspace(-math.pi/2, math.pi/2, self.n), columns=["a0"])
        self.DTpdf["p"] = self.c / (self.c + (1 - self.c) * np.sin(self.DTpdf["a0"])**2)
        self.DTpdf.plot(kind = 'line', x = 'a0', y = 'p', color = 'green')
        plt.show()
        return self.DTpdf

# Representative angle of a line
    def angle_stiefel(self, x, y):
        """
        param x,y Stiefel representation of a line in plane, 
        vectors of the same length
        
        return a vector of angles, [0, pi]
    
        """
        self.alpha = np.arccos(x * np.sign(y))
        return self.alpha

# @param c Relative variance [0,1]
# @param p Probability of predictive interval, (0, 1),
#          defaults to 0.683 to mimic one standard deviation (68.3%)
    def alpha_pred_radius(self, c, p = 0.683):
        """
        Parameters
        c: Relative variance [0,1]
        p: Probability of predictive interval, (0, 1),
        defaults to 0.683 to mimic one standard deviation (68.3%)

        Returns: radius of the p-th predictive interval of subspace angle.
        
        """
        return np.arctan(np.sqrt(c) * np.tan(p/2*np.pi))

#-----------------------------------------------------------------------------------------
#n_new = 101

# Observation points and target points
# 6 points
# theta = c(1/3, 1/2, 3/4, 7/6, 3/2, 5/3, 11/6) * pi
# 7 equi-distance points within 0 and 2 pi

    def set_sample(self):
        self.theta_array = np.array([np.linspace(0.2, 1.8, 7) * math.pi])
        self.theta_array = np.ndarray.round(self.theta_array,7)
        print("theta_array", self.theta_array.T)
        self.theta = self.theta_array.T.tolist()
# Sample size: number of observations
        self.l = len(self.theta)
        print("l:", self.l)
        return self.theta    

# Test points excluding training points
    def set_test_points(self):
        self.thetanew = np.linspace(0, 2, self.n_new) * math.pi
        self.thetanew = np.array(np.setdiff1d(np.ndarray.round(self.thetanew,7),
                                 np.ndarray.round(self.theta_array, 7))) #ask Prof. Zhang on the round
        self.thetanew = self.thetanew.T.tolist()
        return self.thetanew

# Observations: angle (a covering)
    def set_basis(self):
        self.alpha = self.theta_array
        print("alpha\n", self.alpha)
# Observations: Stiefel representation
        self.X = np.vstack((np.cos(self.alpha), np.sin(self.alpha))) #confirm row-bind method employed
        self.X = np.ndarray.round(self.X,7)
        # self.X1 = self.X.reshape(2,7)
        self.X = self.X.tolist()
        return self.X
        # omiited line of r-code to convert to dgeMatrix.

# Hyperparameters
# scale parameter, sigma_p^2, defaults to 1
# sp2 = 1
# lengthscale parameter
# len = pi*0.5
# 6 point optimal (see below)
# NOTE: the special form of this mapping, i.e. linear rotation,
# makes hyperparameter selection difficult for both LOOCV pdf and LOOCV error.
# Occasionally (for six point case), LOOCV pdf becomes useful for this problem.
# With other mappings on G_{1,2}, LOOCV error works in general.

    def set_length_scale(self):
        self.len = (np.amax(self.theta) - np.amin(self.theta)) * 0.6
# Because the true mapping is geodesic, larger correlation length means better prediction.
        self.len = (np.amax(self.theta) - np.amin(self.theta)) * 2
        self.lengthscale = [self.len]
        return self.lengthscale

    def principal(self, i):
        print({
            "i": i, 
            "thetanew[i]": [self.thetanew[i]],
            "theta": self.theta,
            "lengthscale": self.lengthscale,
            "X": self.X
            })
        self.ret = Prediction.GPS_Prediction(self.X, self.theta, [[self.thetanew[i]]], [3], t=None)
        print("Self.ret: \n", self.ret)
        v_list = self.ret[0]
        v = [item[1] for item in v_list]
        alpha = G12.angle_stiefel(self,v[0], v[1])
        lamda = self.ret[1] + self.ret[2]
        print("LAMBDA: ", lamda)
        c = lamda[0]/lamda[1]
        print("c", c)
        data = [[alpha, c]]
        return pd.DataFrame(data, columns=["alpha", "c"])
    
    def DTpred(self):
        a = []
        for i in range(len(self.thetanew)):
            a.append(G12.principal(self, i))
        self.DTpred = pd.concat(a)
        radius = []
        for k in self.DTpred["c"]:
            print("k", k)
            radius.append(self.alpha_pred_radius(k, p=0.95))
        print('K-101', k)
        self.DTpred["radius"] = radius
        self.DTpred["theta"] = self.thetanew
        #compute DT_Train
        self.theta_1 = np.array(np.linspace(0.2, 1.8, 7) * math.pi)
        self.alpha_1 = [i % np.pi for i in self.theta_1]
        radius = [0]*len(self.theta)
        c = [0]*len(self.theta)
        data_train = {'alpha': self.alpha_1,
                 'c': c,
                 'radius': radius,
                 'theta': self.theta_1
                }
        self.DTtrain = pd.DataFrame(data_train)
        print(self.DTpred)
        print(self.DTtrain)
        self.DTpred = np.vstack(((self.DTpred), (self.DTtrain)))
        self.DTpred_table = pd.DataFrame(self.DTpred, columns=["alpha", "c", "radius", "theta"])
        self.DTpred_table = self.DTpred_table.sort_values(by = ['theta'])
        return self.DTpred_table
    
    def curve(self):
        self.alpha_true = [x % np.pi for x in self.thetanew]
        curve_data = {'alpha': self.alpha_true,
                      'theta': self.thetanew
                      }
        self.DT_curve = pd.DataFrame(curve_data)
        return self.DT_curve
        
    def DTpred_plot(self):
        points = {"theta": self.theta_1,
                  "alpha": self.alpha_1
                  }
        self.points = pd.DataFrame(points)
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes()
        columns = ["alpha", "c", "radius", "theta"]
        plt.ylim([0.,np.pi])
        ax.plot(self.DTpred_table['theta'], self.DTpred_table['alpha'], color='black')
        ax.scatter(self.points['theta'], self.points['alpha'], color = 'black')
        ax.plot(self.DT_curve['theta'], self.DT_curve['alpha'], color = 'blue')
        ax.plot(self.DTpred_table['theta'], self.DTpred_table['alpha'] + self.DTpred_table['radius'], color='red')
        ax.plot(self.DTpred_table['theta'], self.DTpred_table['alpha'] - self.DTpred_table['radius'], color='red')
        ax.plot()
        ax.set_xlabel('theta')
        ax.set_ylabel('alpha')
        return plt.show()
    
   
if __name__ == "__main__":
    g12 = G12(
        n = 201,
        c = 0.05,
        n_new = 101,
        )

    # Test set_pdf()
    print("set_pdf:\n", g12.set_pdf())

    # Test set_sample()
    print("set_sample:\n theta: \n", g12.set_sample())

    # Test set_test_points()
    print("set_test_points:\n thetanew: \n", g12.set_test_points())

    # Test set_basis()
    print("set_basis:\n X: \n", g12.set_basis())

    # Test set_length_scale()
    print("set_length_scale:\n", g12.set_length_scale())

    #Test principal()
    print("principal:\n", g12.principal(65))
    
    #Test DTpred()
    print("DTpred:\n", g12.DTpred())
    
    #Test true_prediction()
    print("curve:\n", g12.curve())
    
    #Test DTpred_plot()
    print("DTpred_plot:\n", g12.DTpred_plot())
