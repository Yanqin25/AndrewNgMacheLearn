import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, X, y, lambda1):
    '''Compute cost and gradient for logistic regression with regularization'''
# J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
# theta as the parameter for regularized logistic regression and the
# gradient of the cost w.r.t. to the parameters.

# Initialize some useful values
    m = y.size  # number of training examples

# You need to return the following variables correctly
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

    J = (-np.log(sigmoid(X.dot(theta))).dot(y)-np.log(1-sigmoid(X.dot(theta))
                                                      ).dot(1-y))/m+lambda1/(2*m)*(theta[1:].dot(theta[1:].T))
    return J
# =============================================================


def gradientFunctionReg(theta, X, y, lambda1):
    m = y.size
    grad = (sigmoid(X.dot(theta))-y).dot(X)/m
    temp2 = np.r_[np.zeros(1), lambda1/m*theta[1:]]
    grad = grad+temp2
    return grad


if __name__ == "__main__":
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.c_[np.ones(5), np.arange(1, 16).reshape(5, 3)/10]
    y_t = np.array([1, 0, 1, 0, 1])
    lambda_t = 3
    print(costFunctionReg(theta_t, X_t, y_t, lambda_t))
    print(gradientFunctionReg(theta_t, X_t, y_t, lambda_t))
    print(sigmoid(X_t.dot(theta_t)))
