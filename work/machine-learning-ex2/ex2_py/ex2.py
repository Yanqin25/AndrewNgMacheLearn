from sklearn import linear_model
from predict import predict
from sigmoid import sigmoid
import numpy as np
from plotData import plotData
import matplotlib.pyplot as plt
from costAndGradFunction import costFunction, gradientFunction
from scipy.optimize import minimize
from plotDecisionBoundary import plotDecisionBoundary
# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization


# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=',')
X, y = data[:, :2], data[:, 2]

# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'])


plotData(X, y)

# Put some labels
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(['Admitted', 'Not admitted'], loc='upper right',
           shadow=True, fontsize='x-large', numpoints=1)
# plt.show()
input('\nProgram paused. Press enter to continue.\n')


# ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.py

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradientFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): #f\n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' #f \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = costFunction(test_theta, X, y)
grad = gradientFunction(test_theta, X, y)

print('\nCost at test theta: %f\n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' %f \n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('\nProgram paused. Press enter to continue.\n')


# ============= Part 3: Optimizing using fminunc(scipy for python)   =============
#  In this exercise, you will use scipy  to find the
#  optimal parameters theta.

#  Set options for fminunc

res = minimize(costFunction, initial_theta, method='tnc',
               jac=gradientFunction, args=(X, y), options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})
theta = res.x
cost = res.fun

# Print theta to screen
print('Cost at theta found by fminunc: #f\n', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' #f \n', theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels

# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'], loc='upper right',
           shadow=True, fontsize='x-large', numpoints=1)

# plt.show()
input('\nProgram paused. Press enter to continue.\n')


# ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.py

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.4f\n' % (np.mean(y == p) * 100))
print('Expected accuracy (approx): 89.0\n')
print('\n')

# ============== Part 5: sklearn and Accuracies ==============

regr = linear_model.LogisticRegression(fit_intercept=False)
regr.fit(X, y)
print('theta by sklearn:', regr.coef_.ravel())
print('cost:', costFunction(regr.coef_.flatten(), X, y))
p = predict(regr.coef_.ravel(), X)
print('predict:', np.mean(y == p)*100)
