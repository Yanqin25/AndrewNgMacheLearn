from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn
from plotData import plotData
from warmUpExercise import warmUpExercise
from sklearn import linear_model
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use, cm
# use('TkAgg')

# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
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
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#


# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
warmup = warmUpExercise()
print(warmup)

input('Program paused. Press enter to continue.\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]  # number of training examples
X, y = data[:, 0], data[:, 1]

# Plot Data
# Note: You have to complete the code in plotData.py
plotData(X, y)

input('Program paused. Press enter to continue.\n')


# =================== Part 3: Cost and Gradient descent ===================

X = np.c_[np.ones(m), X]  # Add a column of ones to x
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0  0]\nCost computed = #f\n', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([-1, 2]))
print('\nWith theta = [-1  2]\nCost computed = #f\n', J)
print('Expected cost value (approx) 54.24\n')

input('Program paused. Press enter to continue.\n')
print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, j_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('#f\n', theta)
print('cost:{}'.format(computeCost(X, y, theta)))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
# keep previous plot visible
plt.plot(X[:, 1], np.dot(X, theta), '-', label='Linear regression')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
# don't overlay any more plots on this figure
# plt.show()
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of #f\n', predict1*10000)
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of #f\n', predict2*10000)

input('Program paused. Press enter to continue.\n')


# ## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, X.shape[0])
theta1_vals = np.linspace(-1, 4, X.shape[0])

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, y, t)

#     print(J_vals)
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped

J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(theta0_vals, theta1_vals, J_vals,
                rstride=8, cstride=8, alpha=0.5, cmap='coolwarm', linewidth=10, antialiased=False)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')

# Contour plot
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.clabel(ax, inline=1, fontsize=10)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)



# =============Use Scikit-learn =============
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
# regr = linear_model.LinearRegression()
regr.fit(X, y)
print('theta found by scikit')
print('{} {} \n'.format(regr.coef_[0], regr.coef_[1]))
print('cost:{}'.format(computeCost(X, y, regr.coef_)))
predict1 = np.array([1, 3.5]).dot(regr.coef_)
predict2 = np.array([1, 7]).dot(regr.coef_)
print('For population = 35,000, we predict a profit of {:.4f}'.format(
    predict1*10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(
    predict2*10000))
plotData(X[:, 1], y)
plt.plot(X[:, 1], X.dot(regr.coef_), '-', color='black',
         label='Linear regression wit scikit')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()


# =============Use Normal Equations  =============
data2=np.loadtxt('ex1data1.txt',delimiter=',')
X,y=data2[:,0],data2[:,1]
m=y.size
X=np.c_[np.ones(m),X]
theta=normalEqn(X,y)

print('Theta computed from the normal equations: \n')
print('theta found by normal equations')
print( theta)
print('cost:{}'.format(computeCost(X, y, theta)))
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 35,000, we predict a profit of {:.4f}'.format(
    predict1*10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(
    predict2*10000))

# ============= summry  =============
# linear_model.LinearRegression and normal equations can get the same result