import numpy as np

from sigmoid import sigmoid


def predictOneVsAll(theta, X):
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    res = sigmoid(X.dot(theta))

    pre = np.argmax(res, axis=1)+1
    return pre


if __name__ == "__main__":
    a = np.array([[3, 4, 6], [8, 22, 4]])
    print(np.argmax(a, axis=1)+1)
    print(np.zeros((2, 1)).shape)
