import numpy as np
def computeCost(X, y, theta):
    ''' computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y'''

# Initialize some useful values
    m = len(y); # number of training examples
    J=0
# You need to return the following variables correctly     

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    J=np.sum((np.dot(X,theta)-y)**2)/(2*m)
    return J
# =========================================================================