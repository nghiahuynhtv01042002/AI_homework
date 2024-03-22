import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
def print_data(X0,X1,X,y):
    print("X0= ",X0)
    print("X1 = ",X1)
    print("X = ",X)
    print("y = ",y)

def initial_random():
    np.random.seed(2)

    means = [[2, 2], [4, 2]]
    cov = [[.3, .2], [.2, .3]]
    N = 10

    X0 = np.random.multivariate_normal(means[0], cov, N).T
    X1 = np.random.multivariate_normal(means[1], cov, N).T

    X = np.concatenate((X0, X1), axis = 1)
    y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
    # Xbar 
    X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)
    print_data(X0,X1,X,y)
    print("inital data has finished")
   
    return (X0,X1,X,y) 
