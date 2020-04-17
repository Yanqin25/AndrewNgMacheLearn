import numpy as np
from computeCostMulti import computeCostMulti
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    '''updates theta by taking num_iters gradient steps with learning rate alpha'''
# Initialize some useful values
    m = y.size# number of training examples
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
# ====================== YOUR CODE HERE ======================
# Instructions: Perform a single gradient step on the parameter vector
#               theta. 
#
# Hint: While debugging, it can be useful to print out the values
#       of the cost function (computeCostMulti) and gradient here.
#
        theta=theta-alpha*(X.dot(theta)-y).dot(X)/m
        # ============================================================
        # Save the cost J in every iteration    
        J_history[iter] = computeCostMulti(X, y, theta)
    return theta,J_history


