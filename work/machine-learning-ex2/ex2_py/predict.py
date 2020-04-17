from sigmoid import sigmoid


def predict(theta, X):
    '''Predict whether the label is 0 or 1 using learned logistic regression parameters theta'''
#   p = PREDICT(theta, X) computes the predictions for X using a
#   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

# You need to return the following variables correctly

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#

    p = sigmoid(X.dot(theta)) >= 0.5
    return p
# ========================================================================