
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import initial_random_data as init_data
# generate data
# list of points 
X0,X1,X,y = init_data.initial_random()


import perceptron_function as func

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, miss_point) = func.perceptron(X, y, w_init)
print("mispoint: ",miss_point,end= '\n')
# print(w)
# print(len(w))


## Visualization
import perceptron_animation
import matplotlib.animation 
    
perceptron_animation.viz_alg_1d_2(w,miss_point,X0,X1,X)
