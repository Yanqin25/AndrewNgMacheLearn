import numpy as np


def mapFeature(X1, X2):
    '''Feature mapping function to polynomial features'''
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
#
    out = np.ones(X1.size)
    degree = 7
    for i in range(1, degree):
        for j in range(i+1):
            out = np.c_[out, X1**(i-j)*(X2**j)]
    return out


if __name__ == "__main__":
    print(mapFeature(np.array([2]), np.array([3])))
