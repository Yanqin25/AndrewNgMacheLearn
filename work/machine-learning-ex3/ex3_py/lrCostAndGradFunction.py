import numpy as np

from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lambda1):
    m = y.size
    temp = sigmoid(X.dot(theta))
    j = (-y.dot(np.log(temp)) -
         (1-y).dot(np.log(1-temp)))/m + \
        lambda1/(2*m)*np.sum(theta[1:]**2)
    return j


def lrGradFunction(theta, X, y, lambda1):
    m = y.size

    grad = (sigmoid(X.dot(theta))-y).dot(X)/m+lambda1/m*theta
    grad[0] = (sigmoid(X.dot(theta))-y).dot(X[:, 0])/m
    return grad


if __name__ == "__main__":
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.c_[np.ones(5), np.arange(1, 16).reshape(3, 5).T/10]
    y_t = np.array([1, 0, 1, 0, 1])
    print(X_t, theta_t, y_t)
    lambda_t = 3
    print(lrCostFunction(theta_t, X_t, y_t, lambda_t))

    # print(lrGradFunction(theta_t, X_t, y_t, lambda_t))
    # print(sigmoid(X_t.dot(theta_t)))
    # print(theta_t.shape == (theta_t.size,))
    # print(theta_t.dot(theta_t))
