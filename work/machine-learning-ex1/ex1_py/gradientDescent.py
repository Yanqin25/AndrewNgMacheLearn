import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    '''updates theta by taking num_iters gradient steps with learning rate alpha'''
# Initialize some useful values


    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta.
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCost) and gradient here.
    #
    m = len(y)  # number of training examples
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta = theta-alpha*np.dot(np.dot(X, theta)-y, X)/m
        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history

# ============================================================
if __name__ == "__main__":
    theta=np.array([[1,2,3,4],[4,5,6,7]])
    theta2=np.array([4,5,6,7])
    print(theta@theta2)
    print(theta.dot(theta2))