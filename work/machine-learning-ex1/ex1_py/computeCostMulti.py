import numpy as np
def computeCostMulti(X, y, theta):
#COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
#   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

# Initialize some useful values
    m = y.size # number of training examples

# You need to return the following variables correctly 

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    # J = sum((X*theta-y)'*(X*theta-y))/(2*m)

    J=np.sum((X.dot(theta)-y).dot((X.dot(theta)-y).T))/(2*m)
    return J
# =========================================================================
