import numpy as np
from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from computeCostMulti import computeCostMulti
from normalEqn import normalEqn
import matplotlib.pyplot as plt
## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization

## ================ Part 1: Feature Normalization ================

## Clear and Close Figures

print('Loading data ...\n')

## Load Data
data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Print out some data points
print('First 10 examples from the dataset: \n')
print(' x = {}, y = {} \n' .format(X[:10,:],y[:10]))

input('Program d. Press enter to continue.\n')


#Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = featureNormalize(X)


# Add intercept term to X
X = np.c_[np.ones(m),X]

## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.1
num_iters = 50

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print('cost: {}\n'.format(computeCostMulti(X,y,theta)))
print('Theta computed from gradient descent: \n')
print(' #f \n', theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

 # You should change this

price=np.r_[np.ones(1),(np.array([1650,3])-mu)/sigma].dot(theta)

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):${}\n'.format(price))

input('Program d. Press enter to continue.\n')


## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.py
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Add intercept term to X
X = np.c_[np.ones(m),X]

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('cost: {}\n'.format(computeCostMulti(X,y,theta)))
print('Theta computed from the normal equations: \n')
print(' #f \n', theta)
print('\n')


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# You should change this
price = np.array([1,1650,3]).dot(theta)

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n ${}\n'.format(price))

## ================ Part 4: Use Scikit-learn ================
from sklearn import linear_model

data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = y.size

X=np.c_[np.ones(m),X]
regr=linear_model.LinearRegression(fit_intercept=False,normalize=True)
regr.fit(X,y)

print('cost: {}\n'.format(computeCostMulti(X,y,regr.coef_)))
print('Theta computed from the Scikit-learn: \n')
print(regr.coef_)
print('\n')

price=np.array([1,1650,3]).dot(regr.coef_)
print('Predicted price of a 1650 sq-ft, 3 br house (using the Scikit-learn):\n ${}\n'.format(price))

# ============= summry  =============
# linear_model.LinearRegression and normal equations can get the same result
plt.show()