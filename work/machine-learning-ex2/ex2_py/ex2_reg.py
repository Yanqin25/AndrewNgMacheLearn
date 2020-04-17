
from predict import predict
from plotDecisionBoundary import plotDecisionBoundary
from scipy.optimize import minimize
from mapFeature import mapFeature
from costAndGradFunctionReg import costFunctionReg, gradientFunctionReg
from plotData import plotData
import matplotlib.pyplot as plt
import numpy as np
# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt('ex2data2.txt', delimiter=',')
X, y = data[:, :2], data[:, 2]
plotData(X, y)

# Put some labels
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'], loc='upper right',
           shadow=True, fontsize='small', numpoints=1)
# plt.show()

# =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lambda1 = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
# [cost, grad] = costFunctionReg(initial_theta, X, y, lambda)
cost = costFunctionReg(initial_theta, X, y, lambda1)
grad = gradientFunctionReg(initial_theta, X, y, lambda1)
print('Cost at initial theta (zeros): %f\n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' grad:\n', grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

print('\nProgram paused. Press enter to continue.\n')
# pause

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
# [cost, grad] = costFunctionReg(test_theta, X, y, 10)
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradientFunctionReg(test_theta, X, y, 10)
print('\nCost at test theta (with lambda = 10): %f\n' % cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' grad \n', grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

print('\nProgram paused. Press enter to continue.\n')
# ============= Part 2: Regularization and Accuracies  use scipy.optimize instead of fminunc=============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# # Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])
print(initial_theta)
# Set regularization parameter lambda to 1 (you should vary this)
lambda2 = 1
res = minimize(costFunctionReg, initial_theta, args=(X, y, lambda2),
               method='tnc', jac=gradientFunctionReg, options={'gtol': 1e-3, 'disp': True, 'maxiter': 400})
theta = res.x
cost = res.fun

# # Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.title('lambda = %f' % lambda2)

# # Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'],
           loc='upper right', fontsize='small', shadow=True, numpoints=1)

# # Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %f\n' % (np.mean(y == p) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
plt.show()

